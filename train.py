"""
SWN-MMCF training and ONNX export.

This file contains:
1. Stage-6 encoder-decoder network;
2. robust feature normalization;
3. reconstruction loss;
4. physics-informed dynamic loss;
5. convergence loss;
6. training loop;
7. checkpoint saving;
8. inference-only ONNX export.

The numerical allocation of the three loss weights is intentionally
kept outside the public implementation.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Network configuration
# ============================================================

@dataclass(frozen=True)
class NetworkSpec:
    hidden_dims: tuple[int, ...]
    decoder_dims: tuple[int, ...]
    dropout_after: tuple[bool, ...]
    dropout: float
    negative_slope: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NetworkSpec":
        spec = cls(
            hidden_dims=tuple(int(v) for v in value["hidden_dims"]),
            decoder_dims=tuple(int(v) for v in value["decoder_dims"]),
            dropout_after=tuple(bool(v) for v in value["dropout_after"]),
            dropout=float(value["dropout"]),
            negative_slope=float(value["negative_slope"]),
        )
        if not spec.hidden_dims or any(v <= 0 for v in spec.hidden_dims):
            raise ValueError("hidden_dims must contain positive values")
        if any(v <= 0 for v in spec.decoder_dims):
            raise ValueError("decoder_dims must contain positive values")
        if len(spec.dropout_after) != len(spec.hidden_dims):
            raise ValueError("dropout_after must match hidden_dims")
        if not 0.0 <= spec.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if spec.negative_slope <= 0.0:
            raise ValueError("negative_slope must be positive")
        return spec


# ============================================================
# Robust normalization
# ============================================================

class RobustNormalizer(nn.Module):
    """Median/IQR normalization used identically in training and ONNX."""

    def __init__(self, median: Tensor, scale: Tensor) -> None:
        super().__init__()
        if median.ndim != 1 or scale.ndim != 1:
            raise ValueError("median and scale must be one-dimensional")
        if median.shape != scale.shape:
            raise ValueError("median and scale must have the same shape")
        if not torch.isfinite(median).all() or not torch.isfinite(scale).all():
            raise ValueError("normalizer contains non-finite values")
        if torch.any(scale <= 0):
            raise ValueError("normalizer scale must be positive")
        self.register_buffer("median", median.detach().float())
        self.register_buffer("scale", scale.detach().float())

    def forward(self, feature: Tensor) -> Tensor:
        if feature.ndim != 2 or feature.shape[1] != self.median.numel():
            raise ValueError("unexpected Stage-6 feature dimension")
        return (feature - self.median) / self.scale


# ============================================================
# MLP builder
# ============================================================

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

        layers.append(nn.LeakyReLU(negative_slope=negative_slope))

        if use_dropout and dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        previous = width

    return nn.Sequential(*layers), previous


# ============================================================
# Encoder
# ============================================================

class WeightEncoder(nn.Module):
    """
    Maps normalized measurement-related information to MMCF weights.

    Network:
        d_I -> 1024 -> 512 -> 256 -> N
    when the corresponding values are used in config.
    """

    def __init__(
        self,
        input_dim: int,
        kernel_count: int,
        spec: NetworkSpec,
    ) -> None:

        super().__init__()

        self.backbone, last_dim = dense_stack(
            input_dim,
            spec.hidden_dims,
            batch_norm=True,
            dropout=spec.dropout,
            negative_slope=spec.negative_slope,
            dropout_after=spec.dropout_after,
        )

        self.output = nn.Sequential(
            nn.Linear(last_dim, kernel_count),
            nn.Sigmoid(),
        )

    def forward(self, feature: Tensor) -> Tensor:
        alpha = self.output(self.backbone(feature))

        denominator = alpha.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(torch.finfo(alpha.dtype).tiny)

        return alpha / denominator


# ============================================================
# Decoder
# ============================================================

class PhysicsDecoder(nn.Module):
    """
    Self-supervised reconstruction branch.

    The decoder is only required during training and is removed
    from the inference graph exported to OpenVINS.
    """

    def __init__(
        self,
        kernel_count: int,
        target_dim: int,
        spec: NetworkSpec,
    ) -> None:

        super().__init__()

        self.backbone, last_dim = dense_stack(
            kernel_count,
            spec.decoder_dims,
            batch_norm=False,
            dropout=0.0,
            negative_slope=spec.negative_slope,
        )

        self.output = nn.Linear(last_dim, target_dim)

    def forward(self, alpha: Tensor) -> Tensor:
        return self.output(self.backbone(alpha))


# ============================================================
# Complete training network
# ============================================================

class SWNMMCF(nn.Module):

    def __init__(
        self,
        input_dim: int,
        kernel_count: int,
        reconstruction_dim: int,
        spec: NetworkSpec,
        median: Tensor,
        scale: Tensor,
    ) -> None:

        super().__init__()

        self.normalizer = RobustNormalizer(
            median,
            scale,
        )

        self.encoder = WeightEncoder(
            input_dim,
            kernel_count,
            spec,
        )

        self.decoder = PhysicsDecoder(
            kernel_count,
            reconstruction_dim,
            spec,
        )

    def forward(self, raw_feature: Tensor) -> dict[str, Tensor]:
        feature = self.normalizer(raw_feature)
        alpha = self.encoder(feature)
        reconstruction = self.decoder(alpha)

        return {
            "alpha": alpha,
            "reconstruction": reconstruction,
        }

    def deployment_model(self) -> "DeploymentModel":
        return DeploymentModel(
            self.normalizer,
            self.encoder,
        )


# ============================================================
# Inference-only network
# ============================================================

class DeploymentModel(nn.Module):

    def __init__(
        self,
        normalizer: RobustNormalizer,
        encoder: WeightEncoder,
    ) -> None:

        super().__init__()

        self.normalizer = normalizer
        self.encoder = encoder

    def forward(self, raw_feature: Tensor) -> Tensor:
        feature = self.normalizer(raw_feature)
        return self.encoder(feature)


# ============================================================
# Loss 1: reconstruction loss
# ============================================================

def reconstruction_loss(
    reconstruction: Tensor,
    target: Tensor,
) -> Tensor:
    """
    L_rec = 1/B sum ||rho_hat_k - rho_k||_2^2
    """

    if reconstruction.shape != target.shape:
        raise ValueError(
            "reconstruction and reconstruction_target "
            "must have identical shapes"
        )

    per_sample = torch.sum(
        (reconstruction - target).square(),
        dim=1,
    )

    return per_sample.mean()


# ============================================================
# Disturbance metric
# ============================================================

def mahalanobis_metric(
    residual: Tensor,
    R_inv: Tensor,
) -> Tensor:
    """
    m_k = rho_k^T R_k^{-1} rho_k

    residual : [B, m]
    R_inv    : [B, m, m]
    """

    if residual.ndim != 2:
        raise ValueError("residual must have shape [B, m]")

    if R_inv.ndim != 3:
        raise ValueError("R_inv must have shape [B, m, m]")

    if (
        R_inv.shape[0] != residual.shape[0]
        or R_inv.shape[1] != residual.shape[1]
        or R_inv.shape[2] != residual.shape[1]
    ):
        raise ValueError("residual and R_inv dimensions are inconsistent")

    value = torch.einsum(
        "bi,bij,bj->b",
        residual,
        R_inv,
        residual,
    )

    return value.clamp_min(0.0)


# ============================================================
# Loss 2: physics-informed dynamic loss
# ============================================================

def dynamic_loss(
    alpha: Tensor,
    kernel_scales: Tensor,
    maha: Tensor,
) -> Tensor:
    """
    bar_sigma = [sigma_1^2, ..., sigma_N^2]^T

    r_k = (alpha_k^T bar_sigma) * m_k^2

    L_dyn = 1/B sum r_k
    """

    if kernel_scales.ndim != 1:
        raise ValueError("kernel_scales must be one-dimensional")

    if alpha.ndim != 2:
        raise ValueError("alpha must have shape [B, N]")

    if alpha.shape[1] != kernel_scales.numel():
        raise ValueError(
            "kernel weight number does not match kernel scales"
        )

    if maha.ndim != 1 or maha.shape[0] != alpha.shape[0]:
        raise ValueError("maha must contain one value per sample")

    sigma_penalty = kernel_scales.square()

    weighted_bandwidth = torch.sum(
        alpha * sigma_penalty.unsqueeze(0),
        dim=1,
    )

    per_sample = weighted_bandwidth * maha.square()

    return per_sample.mean()


# ============================================================
# Convergence quantity
# ============================================================

def compute_log_cbar(
    alpha: Tensor,
    batch: Mapping[str, Tensor],
) -> Tensor:
    """
    Compute ln(C_bar_k).

    IMPORTANT
    ---------
    This function must contain exactly the final convergence-bound
    expression derived in the manuscript.

    The public GitHub files currently do not contain the complete
    mathematical expression of C_bar_k, so it should not be guessed
    here.

    The implementation must:
        1. use alpha directly;
        2. remain differentiable with respect to alpha;
        3. return one ln(C_bar_k) for every sample.

    Expected output:
        [B]
    """

    if "log_cbar" in batch:
        log_cbar = batch["log_cbar"]

        if log_cbar.ndim == 2 and log_cbar.shape[1] == 1:
            log_cbar = log_cbar.squeeze(1)

        if log_cbar.ndim != 1:
            raise ValueError("log_cbar must have shape [B]")

        return log_cbar

    raise KeyError(
        "The final differentiable expression of ln(C_bar_k) "
        "must be implemented in compute_log_cbar()."
    )


# ============================================================
# Loss 3: convergence loss
# ============================================================

def convergence_loss(
    log_cbar: Tensor,
    maha: Tensor,
) -> Tensor:
    """
    omega_k = m_k^(-2)

    L_conv =
        [1/B sum softplus(ln(C_bar_k)) * omega_k]^2
    """

    if log_cbar.ndim != 1 or maha.ndim != 1:
        raise ValueError(
            "log_cbar and maha must be one-dimensional"
        )

    if log_cbar.shape != maha.shape:
        raise ValueError(
            "log_cbar and maha must have identical shapes"
        )

    eps = torch.finfo(maha.dtype).eps

    omega = maha.clamp_min(eps).pow(-2)

    penalty = F.softplus(log_cbar)

    return torch.mean(
        penalty * omega
    ).square()


# ============================================================
# Total loss
# ============================================================

class LossComposer:
    """
    The numerical allocation among the three objectives is private.

    The public implementation exposes the three objective terms and
    their total composition, but does not disclose the coefficients.
    """

    def __init__(
        self,
        private_weights: Mapping[str, float],
    ) -> None:

        required = (
            "reconstruction",
            "dynamic",
            "convergence",
        )

        missing = [
            key
            for key in required
            if key not in private_weights
        ]

        if missing:
            raise KeyError(
                f"missing private loss weights: {missing}"
            )

        self._weights = {
            key: float(private_weights[key])
            for key in required
        }

    def __call__(
        self,
        l_rec: Tensor,
        l_dyn: Tensor,
        l_conv: Tensor,
    ) -> Tensor:

        return (
            self._weights["reconstruction"] * l_rec
            + self._weights["dynamic"] * l_dyn
            + self._weights["convergence"] * l_conv
        )


# ============================================================
# Dataset
# ============================================================

class StageDataset(Dataset[dict[str, Tensor]]):

    def __init__(self, path: Path) -> None:

        with np.load(
            path,
            allow_pickle=False,
        ) as archive:

            required = (
                "feature",
                "reconstruction_target",
                "residual",
                "R_inv",
            )

            missing = [
                name
                for name in required
                if name not in archive
            ]

            if missing:
                raise KeyError(
                    f"dataset missing arrays: {missing}"
                )

            count = int(
                archive["feature"].shape[0]
            )

            arrays: dict[str, Tensor] = {}

            for name in archive.files:
                value = np.asarray(
                    archive[name]
                )

                if (
                    value.ndim == 0
                    or value.shape[0] != count
                ):
                    continue

                if np.issubdtype(
                    value.dtype,
                    np.number,
                ):
                    arrays[name] = torch.from_numpy(
                        value.astype(
                            np.float32,
                            copy=True,
                        )
                    )

        if arrays["feature"].ndim != 2:
            raise ValueError(
                "feature must have shape [B, D]"
            )

        if (
            arrays["reconstruction_target"].ndim
            != 2
        ):
            raise ValueError(
                "reconstruction_target must be rank 2"
            )

        if count < 2:
            raise ValueError(
                "dataset contains too few samples"
            )

        if any(
            not value.isfinite().all()
            for value in arrays.values()
        ):
            raise ValueError(
                "dataset contains non-finite values"
            )

        self.arrays = arrays
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor]:

        return {
            name: value[index]
            for name, value
            in self.arrays.items()
        }


# ============================================================
# Normalization statistics
# ============================================================

def fit_normalizer(
    feature: Tensor,
    sample_limit: int,
    seed: int,
) -> tuple[Tensor, Tensor]:

    if sample_limit <= 0:
        raise ValueError(
            "scaler_max_samples must be positive"
        )

    generator = torch.Generator().manual_seed(
        seed
    )

    indices = torch.randperm(
        feature.shape[0],
        generator=generator,
    )[:sample_limit]

    sample = feature[indices]

    median = torch.quantile(
        sample,
        0.5,
        dim=0,
    )

    q75 = torch.quantile(
        sample,
        0.75,
        dim=0,
    )

    q25 = torch.quantile(
        sample,
        0.25,
        dim=0,
    )

    scale = q75 - q25

    scale = torch.where(
        scale > torch.finfo(scale.dtype).eps,
        scale,
        torch.ones_like(scale),
    )

    return median, scale


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Training
# ============================================================

def run_training(
    args: argparse.Namespace,
) -> None:

    config = json.loads(
        args.config.read_text(
            encoding="utf-8"
        )
    )

    training = config["training"]
    spec = NetworkSpec.from_mapping(
        config["network"]
    )

    seed = int(training["seed"])
    epochs = int(training["epochs"])
    batch_size = int(
        training["batch_size"]
    )

    if epochs <= 0:
        raise ValueError(
            "epochs must be positive"
        )

    if batch_size <= 1:
        raise ValueError(
            "batch_size must exceed one"
        )

    seed_everything(seed)

    dataset = StageDataset(
        args.data
    )

    input_dim = int(
        dataset.arrays["feature"].shape[1]
    )

    reconstruction_dim = int(
        dataset.arrays[
            "reconstruction_target"
        ].shape[1]
    )

    kernel_scales = torch.tensor(
        training["kernel_scales"],
        dtype=torch.float32,
    )

    kernel_count = int(
        kernel_scales.numel()
    )

    median, scale = fit_normalizer(
        dataset.arrays["feature"],
        min(
            int(
                training[
                    "scaler_max_samples"
                ]
            ),
            len(dataset),
        ),
        seed,
    )

    device = torch.device(
        args.device
        or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    model = SWNMMCF(
        input_dim=input_dim,
        kernel_count=kernel_count,
        reconstruction_dim=reconstruction_dim,
        spec=spec,
        median=median,
        scale=scale,
    ).to(device)

    kernel_scales = kernel_scales.to(
        device
    )

    #
    # Numerical loss coefficients are read from a private file
    # and are intentionally not contained in the public source.
    #
    private_loss_config = json.loads(
        args.private_loss_config.read_text(
            encoding="utf-8"
        )
    )

    compose_total_loss = LossComposer(
        private_loss_config
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            training["learning_rate"]
        ),
        weight_decay=float(
            training["weight_decay"]
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(
            training["num_workers"]
        ),
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=True,
    )

    if len(loader) == 0:
        raise ValueError(
            "batch size exceeds dataset size"
        )

    for epoch in range(epochs):

        model.train()

        epoch_total = 0.0
        epoch_rec = 0.0
        epoch_dyn = 0.0
        epoch_conv = 0.0

        for host_batch in loader:

            batch = {
                name: value.to(
                    device,
                    non_blocking=True,
                )
                for name, value
                in host_batch.items()
            }

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                batch["feature"]
            )

            alpha = outputs["alpha"]

            maha = mahalanobis_metric(
                batch["residual"],
                batch["R_inv"],
            )

            l_rec = reconstruction_loss(
                outputs["reconstruction"],
                batch[
                    "reconstruction_target"
                ],
            )

            l_dyn = dynamic_loss(
                alpha,
                kernel_scales,
                maha,
            )

            log_cbar = compute_log_cbar(
                alpha,
                batch,
            )

            l_conv = convergence_loss(
                log_cbar,
                maha,
            )

            loss = compose_total_loss(
                l_rec,
                l_dyn,
                l_conv,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "non-finite total loss"
                )

            loss.backward()

            optimizer.step()

            epoch_total += float(
                loss.detach()
            )

            epoch_rec += float(
                l_rec.detach()
            )

            epoch_dyn += float(
                l_dyn.detach()
            )

            epoch_conv += float(
                l_conv.detach()
            )

        count = len(loader)

        print(
            f"epoch={epoch + 1:04d} "
            f"loss={epoch_total / count:.6e} "
            f"rec={epoch_rec / count:.6e} "
            f"dyn={epoch_dyn / count:.6e} "
            f"conv={epoch_conv / count:.6e}"
        )

    # ========================================================
    # Save trained checkpoint
    # ========================================================

    args.checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": input_dim,
            "kernel_count": kernel_count,
            "reconstruction_dim": reconstruction_dim,
        },
        args.checkpoint,
    )

    # ========================================================
    # Export inference-only ONNX
    # ========================================================

    args.onnx.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    deployment_model = (
        model.deployment_model()
        .eval()
    )

    example = torch.zeros(
        1,
        input_dim,
        dtype=torch.float32,
        device=device,
    )

    torch.onnx.export(
        deployment_model,
        example,
        args.onnx,
        input_names=["x"],
        output_names=["alpha"],
        dynamic_axes={
            "x": {
                0: "batch",
            },
            "alpha": {
                0: "batch",
            },
        },
        opset_version=int(
            training["onnx_opset"]
        ),
        dynamo=False,
    )


# ============================================================
# Command line
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--private-loss-config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_training(
        parse_arguments()
    )
