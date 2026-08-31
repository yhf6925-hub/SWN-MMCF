"""Core SWN-MMCF Stage-6 training loop with a private objective hook."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from model import NetworkSpec, Stage6TrainingModel


Objective = Callable[
    [Mapping[str, Tensor], Mapping[str, Tensor], int],
    Tensor | tuple[Tensor, Mapping[str, float]],
]


def compose_private_objective(
    outputs: Mapping[str, Tensor], batch: Mapping[str, Tensor], epoch: int
) -> Tensor:
    """Confidential multi-loss composition contract.

    The real implementation is intentionally maintained in an untracked module.
    It must return one finite scalar tensor and may use any batch fields. Keeping
    this single boundary private avoids publishing loss weights or their schedule.
    """
    del outputs, batch, epoch
    raise NotImplementedError(
        "provide --objective package.module:function from a private module"
    )


class NpzTensorDataset(Dataset[dict[str, Tensor]]):
    """Loads aligned numeric arrays; `feature` and `physics_target` are required."""

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as archive:
            if "feature" not in archive or "physics_target" not in archive:
                raise KeyError("dataset must contain feature and physics_target")
            sample_count = int(archive["feature"].shape[0])
            arrays: dict[str, Tensor] = {}
            for key in archive.files:
                value = np.asarray(archive[key])
                if value.ndim == 0 or value.shape[0] != sample_count:
                    continue
                if not np.issubdtype(value.dtype, np.number):
                    raise TypeError(f"dataset field {key!r} is not numeric")
                arrays[key] = torch.from_numpy(value.astype(np.float32, copy=True))

        feature = arrays["feature"]
        target = arrays["physics_target"]
        if feature.ndim != 2 or target.ndim != 2:
            raise ValueError("feature and physics_target must be rank-2 arrays")
        if sample_count < 2:
            raise ValueError("dataset must contain at least two samples")
        if not torch.isfinite(feature).all() or not torch.isfinite(target).all():
            raise ValueError("dataset contains non-finite required values")
        self.arrays = arrays
        self.sample_count = sample_count

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {key: value[index] for key, value in self.arrays.items()}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError("private configuration root must be an object")
    return value


def _require(config: Mapping[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(f"missing private training keys: {', '.join(missing)}")


def _load_objective(reference: str) -> Objective:
    try:
        module_name, function_name = reference.split(":", maxsplit=1)
    except ValueError as error:
        raise ValueError("objective must use package.module:function syntax") from error
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"objective {reference!r} is not callable")
    return function


def _robust_normalizer(
    features: Tensor, maximum_samples: int, seed: int
) -> tuple[Tensor, Tensor]:
    if maximum_samples <= 0:
        raise ValueError("scaler_max_samples must be positive")
    generator = torch.Generator().manual_seed(seed)
    count = min(maximum_samples, features.shape[0])
    indices = torch.randperm(features.shape[0], generator=generator)[:count]
    sample = features[indices]
    median = torch.quantile(sample, 0.5, dim=0)
    lower = torch.quantile(sample, 0.25, dim=0)
    upper = torch.quantile(sample, 0.75, dim=0)
    scale = upper - lower
    scale = torch.where(scale > torch.finfo(scale.dtype).eps, scale, torch.ones_like(scale))
    return median, scale


def _objective_value(
    objective: Objective,
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    epoch: int,
) -> Tensor:
    result = objective(outputs, batch, epoch)
    loss = result[0] if isinstance(result, tuple) else result
    if not isinstance(loss, Tensor) or loss.ndim != 0:
        raise TypeError("private objective must return a scalar Tensor")
    if not torch.isfinite(loss):
        raise FloatingPointError("private objective returned a non-finite loss")
    return loss


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    config = _load_json(args.config)
    training = config.get("training")
    network = config.get("network")
    if not isinstance(training, dict) or not isinstance(network, dict):
        raise KeyError("private config requires training and network objects")
    _require(
        training,
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "seed",
        "num_workers",
        "scaler_max_samples",
        "kernel_count",
    )
    if args.onnx is not None:
        _require(training, "onnx_opset")

    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    seed = int(training["seed"])
    if epochs <= 0 or batch_size <= 1:
        raise ValueError("epochs must be positive and batch_size must exceed one")
    _seed_everything(seed)

    dataset = NpzTensorDataset(args.data)
    median, scale = _robust_normalizer(
        dataset.arrays["feature"], int(training["scaler_max_samples"]), seed
    )
    input_dim = int(dataset.arrays["feature"].shape[1])
    physics_dim = int(dataset.arrays["physics_target"].shape[1])
    kernel_count = int(training["kernel_count"])

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Stage6TrainingModel(
        input_dim=input_dim,
        kernel_count=kernel_count,
        physics_dim=physics_dim,
        spec=NetworkSpec.from_mapping(network),
        median=median,
        scale=scale,
    ).to(device)
    objective = _load_objective(args.objective)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("batch_size is larger than the available dataset")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for host_batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in host_batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["feature"])
            loss = _objective_value(objective, outputs, batch, epoch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())
        print(f"epoch={epoch + 1} objective={running_loss / len(loader):.6e}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": input_dim,
            "kernel_count": kernel_count,
            "physics_dim": physics_dim,
        },
        args.checkpoint,
    )

    if args.onnx is not None:
        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        deployment = model.deployment_model().eval()
        example = torch.zeros(1, input_dim, dtype=torch.float32, device=device)
        torch.onnx.export(
            deployment,
            example,
            args.onnx,
            input_names=["x"],
            output_names=["alpha"],
            dynamic_axes={"x": {0: "batch"}, "alpha": {0: "batch"}},
            opset_version=int(training["onnx_opset"]),
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="private Stage-6 NPZ dump")
    parser.add_argument("--config", type=Path, required=True, help="untracked private JSON config")
    parser.add_argument(
        "--objective",
        required=True,
        help="private loss composer in package.module:function form",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--device", help="optional torch device, for example cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    train(_arguments())
