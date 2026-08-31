"""Compact SWN-MMCF Stage-6 network definition.

The public model contains the inference architecture and normalization path.
Training hyperparameters and the confidential multi-loss policy are deliberately
kept outside this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class NetworkSpec:
    """Architecture values loaded from a private training configuration."""

    hidden_dims: tuple[int, ...]
    decoder_dims: tuple[int, ...]
    dropout_after: tuple[bool, ...]
    dropout: float
    negative_slope: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NetworkSpec":
        required = (
            "hidden_dims",
            "decoder_dims",
            "dropout_after",
            "dropout",
            "negative_slope",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise KeyError(f"missing network configuration keys: {', '.join(missing)}")

        spec = cls(
            hidden_dims=tuple(int(width) for width in value["hidden_dims"]),
            decoder_dims=tuple(int(width) for width in value["decoder_dims"]),
            dropout_after=tuple(bool(enabled) for enabled in value["dropout_after"]),
            dropout=float(value["dropout"]),
            negative_slope=float(value["negative_slope"]),
        )
        if not spec.hidden_dims or any(width <= 0 for width in spec.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        if any(width <= 0 for width in spec.decoder_dims):
            raise ValueError("decoder_dims must contain positive integers")
        if len(spec.dropout_after) != len(spec.hidden_dims):
            raise ValueError("dropout_after must match hidden_dims length")
        if not 0.0 <= spec.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if spec.negative_slope <= 0.0:
            raise ValueError("negative_slope must be positive")
        return spec


class RobustNormalizer(nn.Module):
    """Median/IQR normalization embedded in both training and ONNX export."""

    def __init__(self, median: Tensor, scale: Tensor) -> None:
        super().__init__()
        if median.ndim != 1 or scale.ndim != 1 or median.shape != scale.shape:
            raise ValueError("median and scale must be same-length vectors")
        if not torch.isfinite(median).all() or not torch.isfinite(scale).all():
            raise ValueError("normalizer contains non-finite values")
        if torch.any(scale <= 0):
            raise ValueError("normalizer scale must be strictly positive")
        self.register_buffer("median", median.detach().to(dtype=torch.float32))
        self.register_buffer("scale", scale.detach().to(dtype=torch.float32))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.median.numel():
            raise ValueError("unexpected Stage-6 feature shape")
        return (features - self.median) / self.scale


def _hidden_stack(
    input_dim: int,
    widths: Sequence[int],
    dropout: float,
    negative_slope: float,
    *,
    batch_norm: bool,
    dropout_after: Sequence[bool] | None = None,
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    previous = input_dim
    dropout_mask = tuple(dropout_after or (False,) * len(widths))
    if len(dropout_mask) != len(widths):
        raise ValueError("dropout mask and widths must have equal length")
    for width, use_dropout in zip(widths, dropout_mask, strict=True):
        layers.append(nn.Linear(previous, width))
        if batch_norm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(nn.LeakyReLU(negative_slope=negative_slope))
        if use_dropout and dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        previous = width
    return nn.Sequential(*layers), previous


class Stage6Encoder(nn.Module):
    """Predicts normalized MMCF kernel-mixture coefficients."""

    def __init__(self, input_dim: int, kernel_count: int, spec: NetworkSpec) -> None:
        super().__init__()
        if input_dim <= 0 or kernel_count <= 0:
            raise ValueError("input_dim and kernel_count must be positive")
        self.backbone, last_dim = _hidden_stack(
            input_dim,
            spec.hidden_dims,
            spec.dropout,
            spec.negative_slope,
            batch_norm=True,
            dropout_after=spec.dropout_after,
        )
        self.head = nn.Sequential(nn.Linear(last_dim, kernel_count), nn.Sigmoid())

    def forward(self, features: Tensor) -> Tensor:
        weights = self.head(self.backbone(features))
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        return weights / denominator


class PhysicsDecoder(nn.Module):
    """Training-only decoder used by the private self-supervised objective."""

    def __init__(self, kernel_count: int, output_dim: int, spec: NetworkSpec) -> None:
        super().__init__()
        self.backbone, last_dim = _hidden_stack(
            kernel_count,
            spec.decoder_dims,
            0.0,
            spec.negative_slope,
            batch_norm=False,
        )
        self.head = nn.Linear(last_dim, output_dim)

    def forward(self, weights: Tensor) -> Tensor:
        return self.head(self.backbone(weights))


class Stage6TrainingModel(nn.Module):
    """End-to-end public training surface: raw feature -> alpha + reconstruction."""

    def __init__(
        self,
        input_dim: int,
        kernel_count: int,
        physics_dim: int,
        spec: NetworkSpec,
        median: Tensor,
        scale: Tensor,
    ) -> None:
        super().__init__()
        if physics_dim <= 0:
            raise ValueError("physics_dim must be positive")
        self.normalizer = RobustNormalizer(median, scale)
        self.encoder = Stage6Encoder(input_dim, kernel_count, spec)
        self.decoder = PhysicsDecoder(kernel_count, physics_dim, spec)

    def forward(self, raw_features: Tensor) -> dict[str, Tensor]:
        normalized = self.normalizer(raw_features)
        alpha = self.encoder(normalized)
        return {"alpha": alpha, "physics": self.decoder(alpha)}

    def deployment_model(self) -> "Stage6DeploymentModel":
        return Stage6DeploymentModel(self.normalizer, self.encoder)


class Stage6DeploymentModel(nn.Module):
    """Inference-only graph exported for the OpenVINS ONNX Runtime adapter."""

    def __init__(self, normalizer: RobustNormalizer, encoder: Stage6Encoder) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.encoder = encoder

    def forward(self, raw_features: Tensor) -> Tensor:
        return self.encoder(self.normalizer(raw_features))
