"""Core training pipeline for the SWN-MMCF self-supervised weight network."""

from __future__ import annotations

import argparse
import importlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


LossResult = Tensor | tuple[Tensor, Mapping[str, float]]
LossFunction = Callable[[Mapping[str, Tensor], Mapping[str, Tensor], int], LossResult]


@dataclass(frozen=True)
class NetworkSpec:
    hidden_dims: tuple[int, ...]
    decoder_widths: tuple[int, ...]
    dropout_after: tuple[bool, ...]
    dropout: float
    negative_slope: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NetworkSpec":
        spec = cls(
            hidden_dims=tuple(int(v) for v in value["hidden_dims"]),
            decoder_widths=tuple(int(v) for v in value["decoder_dims"]),
            dropout_after=tuple(bool(v) for v in value["dropout_after"]),
            dropout=float(value["dropout"]),
            negative_slope=float(value["negative_slope"]),
        )
        if not spec.hidden_dims or any(v <= 0 for v in spec.hidden_dims):
            raise ValueError("hidden_dims must contain positive values")
        if any(v <= 0 for v in spec.decoder_widths):
            raise ValueError("decoder_widths must contain positive values")
        if len(spec.dropout_after) != len(spec.hidden_dims):
            raise ValueError("dropout_after must match hidden_dims")
        if not 0.0 <= spec.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if spec.negative_slope <= 0.0:
            raise ValueError("negative_slope must be positive")
        return spec


class RobustNormalizer(nn.Module):
    """Median/IQR normalization shared by training and exported inference."""

    def __init__(self, median: Tensor, scale: Tensor) -> None:
        super().__init__()
        self.register_buffer("median", median.detach().float())
        self.register_buffer("scale", scale.detach().float())

    def forward(self, feature: Tensor) -> Tensor:
        return (feature - self.median) / self.scale


def dense_stack(
    input_dim: int,
    widths: Sequence[int],
    *,
    batch_norm: bool,
    dropout: float,
    negative_slope: float,
    dropout_after: Sequence[bool] | None = None,
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    previous = input_dim
    dropout_mask = tuple(dropout_after or (False,) * len(widths))
    if len(dropout_mask) != len(widths):
        raise ValueError("dropout mask must match layer widths")
    for width, use_dropout in zip(widths, dropout_mask, strict=True):
        layers.append(nn.Linear(previous, width))
        if batch_norm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(nn.LeakyReLU(negative_slope))
        if use_dropout and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        previous = width
    return nn.Sequential(*layers), previous


class WeightEncoder(nn.Module):
    """Maps physical-information features to normalized MMCF kernel weights."""

    def __init__(self, input_dim: int, kernel_count: int, spec: NetworkSpec) -> None:
        super().__init__()
        self.backbone, output_dim = dense_stack(
            input_dim,
            spec.hidden_dims,
            batch_norm=True,
            dropout=spec.dropout,
            negative_slope=spec.negative_slope,
            dropout_after=spec.dropout_after,
        )
        self.output = nn.Sequential(nn.Linear(output_dim, kernel_count), nn.Sigmoid())

    def forward(self, feature: Tensor) -> Tensor:
        alpha = self.output(self.backbone(feature))
        return alpha / alpha.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(alpha.dtype).tiny
        )


class PhysicsDecoder(nn.Module):
    """Training branch that reconstructs the physical-information target."""

    def __init__(self, kernel_count: int, target_dim: int, spec: NetworkSpec) -> None:
        super().__init__()
        self.backbone, output_dim = dense_stack(
            kernel_count,
            spec.decoder_widths,
            batch_norm=False,
            dropout=0.0,
            negative_slope=spec.negative_slope,
        )
        self.output = nn.Linear(output_dim, target_dim)

    def forward(self, alpha: Tensor) -> Tensor:
        return self.output(self.backbone(alpha))


class TrainingGraph(nn.Module):
    def __init__(
        self,
        input_dim: int,
        kernel_count: int,
        target_dim: int,
        spec: NetworkSpec,
        median: Tensor,
        scale: Tensor,
    ) -> None:
        super().__init__()
        self.normalizer = RobustNormalizer(median, scale)
        self.encoder = WeightEncoder(input_dim, kernel_count, spec)
        self.decoder = PhysicsDecoder(kernel_count, target_dim, spec)

    def forward(self, raw_feature: Tensor) -> dict[str, Tensor]:
        normalized = self.normalizer(raw_feature)
        alpha = self.encoder(normalized)
        return {"alpha": alpha, "physics": self.decoder(alpha)}

    def inference_graph(self) -> "InferenceGraph":
        return InferenceGraph(self.normalizer, self.encoder)


class InferenceGraph(nn.Module):
    """The raw-feature to robust-weight graph exported to OpenVINS."""

    def __init__(self, normalizer: RobustNormalizer, encoder: WeightEncoder) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.encoder = encoder

    def forward(self, raw_feature: Tensor) -> Tensor:
        return self.encoder(self.normalizer(raw_feature))


class StageDataset(Dataset[dict[str, Tensor]]):
    """Loads aligned physical-information arrays from a numeric NPZ archive."""

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as archive:
            if "feature" not in archive or "physics_target" not in archive:
                raise KeyError("feature and physics_target are required")
            count = int(archive["feature"].shape[0])
            arrays: dict[str, Tensor] = {}
            for name in archive.files:
                value = np.asarray(archive[name])
                if value.ndim == 0 or value.shape[0] != count:
                    continue
                if np.issubdtype(value.dtype, np.number):
                    arrays[name] = torch.from_numpy(value.astype(np.float32, copy=True))

        if arrays["feature"].ndim != 2 or arrays["physics_target"].ndim != 2:
            raise ValueError("feature and physics_target must be rank-2 arrays")
        if count < 2 or any(not torch.isfinite(value).all() for value in arrays.values()):
            raise ValueError("dataset must contain finite aligned samples")
        self.arrays = arrays
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {name: value[index] for name, value in self.arrays.items()}


def fit_normalizer(feature: Tensor, sample_limit: int, seed: int) -> tuple[Tensor, Tensor]:
    if sample_limit <= 0:
        raise ValueError("scaler_max_samples must be positive")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(feature.shape[0], generator=generator)[:sample_limit]
    sample = feature[indices]
    median = torch.quantile(sample, 0.5, dim=0)
    scale = torch.quantile(sample, 0.75, dim=0) - torch.quantile(sample, 0.25, dim=0)
    scale = torch.where(scale > torch.finfo(scale.dtype).eps, scale, torch.ones_like(scale))
    return median, scale


def load_loss_function(reference: str) -> LossFunction:
    module_name, function_name = reference.split(":", maxsplit=1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{reference!r} is not callable")
    return function


def compose_loss(
    loss_function: LossFunction,
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    epoch: int,
) -> Tensor:
    """Compose the reconstruction, physical and convergence objectives."""

    result = loss_function(outputs, batch, epoch)
    loss = result[0] if isinstance(result, tuple) else result
    if not isinstance(loss, Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("loss function must return one finite scalar tensor")
    return loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    spec = NetworkSpec.from_mapping(config["network"])
    seed = int(training["seed"])
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    if epochs <= 0 or batch_size <= 1:
        raise ValueError("epochs must be positive and batch_size must exceed one")
    seed_everything(seed)

    dataset = StageDataset(args.data)
    input_dim = int(dataset.arrays["feature"].shape[1])
    target_dim = int(dataset.arrays["physics_target"].shape[1])
    kernel_count = int(training["kernel_count"])
    median, scale = fit_normalizer(
        dataset.arrays["feature"],
        min(int(training["scaler_max_samples"]), len(dataset)),
        seed,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TrainingGraph(
        input_dim, kernel_count, target_dim, spec, median, scale
    ).to(device)
    loss_function = load_loss_function(args.loss)
    optimizer = torch.optim.Adam(
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
        raise ValueError("batch size exceeds the available training samples")

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for host_batch in loader:
            batch = {
                name: value.to(device, non_blocking=True)
                for name, value in host_batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["feature"])
            loss = compose_loss(loss_function, outputs, batch, epoch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        print(f"epoch={epoch + 1} loss={total / len(loader):.6e}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": input_dim,
            "kernel_count": kernel_count,
            "target_dim": target_dim,
        },
        args.checkpoint,
    )

    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    inference = model.inference_graph().eval()
    example = torch.zeros(1, input_dim, dtype=torch.float32, device=device)
    torch.onnx.export(
        inference,
        example,
        args.onnx,
        input_names=["x"],
        output_names=["alpha"],
        dynamic_axes={"x": {0: "batch"}, "alpha": {0: "batch"}},
        opset_version=int(training["onnx_opset"]),
        dynamo=False,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--loss", required=True, help="package.module:function")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--device")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_arguments())
