"""SWN-MMCF network training, self-supervised objectives and ONNX export."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class MmcfSpec:
    q: float
    sigma_floor: float
    contraction_target: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MmcfSpec":
        spec = cls(
            q=float(value["q"]),
            sigma_floor=float(value["sigma_floor"]),
            contraction_target=float(value["contraction_target"]),
        )
        if not 1.0 / 3.0 < spec.q < 1.0:
            raise ValueError("q must satisfy 1/3 < q < 1")
        if spec.sigma_floor <= 0.0:
            raise ValueError("sigma_floor must be positive")
        if not 0.0 < spec.contraction_target < 1.0:
            raise ValueError("contraction_target must be in (0, 1)")
        return spec


@dataclass(frozen=True)
class LossWeights:
    reconstruction: float
    dynamic: float
    convergence: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LossWeights":
        weights = cls(
            reconstruction=float(value["reconstruction"]),
            dynamic=float(value["dynamic"]),
            convergence=float(value["convergence"]),
        )
        if any(weight < 0.0 for weight in vars(weights).values()):
            raise ValueError("loss weights must be non-negative")
        return weights


@dataclass(frozen=True)
class LossTerms:
    reconstruction: Tensor
    dynamic: Tensor
    convergence: Tensor


class RobustNormalizer(nn.Module):
    """Median/IQR normalization shared by training and deployment."""

    def __init__(self, median: Tensor, scale: Tensor) -> None:
        super().__init__()
        if median.ndim != 1 or scale.ndim != 1 or median.shape != scale.shape:
            raise ValueError("median and scale must be same-length vectors")
        if not torch.isfinite(median).all() or not torch.isfinite(scale).all():
            raise ValueError("normalizer contains non-finite values")
        if torch.any(scale <= 0.0):
            raise ValueError("normalizer scale must be positive")
        self.register_buffer("median", median.detach().float())
        self.register_buffer("scale", scale.detach().float())

    def forward(self, feature: Tensor) -> Tensor:
        if feature.ndim != 2 or feature.shape[1] != self.median.numel():
            raise ValueError("unexpected Stage-6 feature dimension")
        return (feature - self.median) / self.scale


class WeightEncoder(nn.Module):
    """Maps physical-information features to normalized MMCF weights."""

    def __init__(self, input_dim: int, kernel_count: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, kernel_count),
            nn.Sigmoid(),
        )

    def forward(self, feature: Tensor) -> Tensor:
        alpha = self.net(feature)
        denominator = alpha.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(alpha.dtype).tiny
        )
        return alpha / denominator


class PhysicsDecoder(nn.Module):
    """Training branch that reconstructs the physical target."""

    def __init__(self, kernel_count: int, target_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(kernel_count, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, target_dim),
        )

    def forward(self, alpha: Tensor) -> Tensor:
        return self.net(alpha)


class SWNMMCF(nn.Module):
    def __init__(
        self,
        input_dim: int,
        kernel_count: int,
        target_dim: int,
        median: Tensor,
        scale: Tensor,
    ) -> None:
        super().__init__()
        self.normalizer = RobustNormalizer(median, scale)
        self.encoder = WeightEncoder(input_dim, kernel_count)
        self.decoder = PhysicsDecoder(kernel_count, target_dim)

    def forward(self, raw_feature: Tensor) -> dict[str, Tensor]:
        alpha = self.encoder(self.normalizer(raw_feature))
        return {"alpha": alpha, "reconstruction": self.decoder(alpha)}

    def deployment_model(self) -> "DeploymentModel":
        return DeploymentModel(self.normalizer, self.encoder)


class DeploymentModel(nn.Module):
    def __init__(self, normalizer: RobustNormalizer, encoder: WeightEncoder) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.encoder = encoder

    def forward(self, raw_feature: Tensor) -> Tensor:
        return self.encoder(self.normalizer(raw_feature))


class StageDataset(Dataset[dict[str, Tensor]]):
    """Loads aligned padded Stage-6 arrays from an NPZ archive."""

    REQUIRED = (
        "feature",
        "physics_target",
        "maha",
        "beta_k",
        "P_inv",
        "HtRinvH",
        "H",
        "z",
        "R_diag",
        "x_prior",
        "mask_mea",
        "mask_sta",
    )

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as archive:
            missing = [name for name in self.REQUIRED if name not in archive]
            if missing:
                raise KeyError(f"dataset missing arrays: {', '.join(missing)}")
            count = int(archive["feature"].shape[0])
            arrays: dict[str, Tensor] = {}
            for name in archive.files:
                value = np.asarray(archive[name])
                if value.ndim == 0 or value.shape[0] != count:
                    continue
                if np.issubdtype(value.dtype, np.number):
                    arrays[name] = torch.from_numpy(value.astype(np.float32, copy=True))

        feature = arrays["feature"]
        target = arrays["physics_target"]
        H = arrays["H"]
        z = arrays["z"]
        x_prior = arrays["x_prior"]
        if feature.ndim != 2 or target.ndim != 2:
            raise ValueError("feature and physics_target must be rank-2 arrays")
        if H.ndim != 3 or z.ndim != 2 or x_prior.ndim != 2:
            raise ValueError("H, z and x_prior have invalid ranks")
        if H.shape[1:] != (z.shape[1], x_prior.shape[1]):
            raise ValueError("H, z and x_prior dimensions are inconsistent")
        if arrays["R_diag"].shape != z.shape or arrays["mask_mea"].shape != z.shape:
            raise ValueError("measurement arrays must match z")
        state_shape = (count, x_prior.shape[1], x_prior.shape[1])
        if arrays["P_inv"].shape != state_shape or arrays["HtRinvH"].shape != state_shape:
            raise ValueError("state information matrices have invalid dimensions")
        if arrays["mask_sta"].shape != x_prior.shape:
            raise ValueError("mask_sta must match x_prior")
        beta_k = arrays["beta_k"]
        if beta_k.ndim not in (1, 2) or beta_k.reshape(count, -1).shape[1] != 1:
            raise ValueError("beta_k must contain one convergence radius per sample")
        if torch.any(beta_k < 0.0):
            raise ValueError("beta_k must be non-negative")
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


def _maha_vector(maha: Tensor, batch_size: int) -> Tensor:
    value = maha.reshape(batch_size, -1)[:, 0]
    return value.clamp_min(torch.finfo(value.dtype).eps)


def compute_log_cbar(
    alpha: Tensor,
    kernel_scales: Tensor,
    batch: Mapping[str, Tensor],
    spec: MmcfSpec,
) -> Tensor:
    """Compute the beta-bounded uniform contraction quantity for each sample."""

    batch_size, kernel_count = alpha.shape
    sigma = kernel_scales.reshape(1, kernel_count).expand(batch_size, kernel_count)
    sigma = sigma.clamp_min(spec.sigma_floor)
    log_sigma = torch.log(sigma)
    eps = torch.finfo(alpha.dtype).eps
    log_alpha = torch.log(alpha.clamp_min(eps))
    mask_mea = batch["mask_mea"].to(dtype=alpha.dtype)
    mask_sta = batch["mask_sta"].to(dtype=alpha.dtype)
    state_mask = mask_sta.unsqueeze(1) * mask_sta.unsqueeze(2)
    P_inv = 0.5 * (batch["P_inv"] + batch["P_inv"].transpose(1, 2))
    P_inv = P_inv * state_mask
    HtRinvH = 0.5 * (batch["HtRinvH"] + batch["HtRinvH"].transpose(1, 2))
    HtRinvH = HtRinvH * state_mask
    H = batch["H"] * mask_mea.unsqueeze(2) * mask_sta.unsqueeze(1)
    R_inv = mask_mea / batch["R_diag"].clamp_min(eps)
    sqrt_R_inv = torch.sqrt(R_inv)
    beta_k = batch["beta_k"].reshape(batch_size, -1)[:, 0].clamp_min(0.0)

    rho_k = (
        batch["z"] - torch.bmm(H, batch["x_prior"].unsqueeze(-1)).squeeze(-1)
    ) * mask_mea
    weighted_rho = sqrt_R_inv * rho_k
    weighted_H = sqrt_R_inv.unsqueeze(2) * H

    prior_spectrum = torch.linalg.eigvalsh(P_inv)
    prior_norm = prior_spectrum[:, -1].clamp_min(eps)
    weighted_H_norm = torch.linalg.matrix_norm(weighted_H, ord=2)
    prior_distance_bound = (prior_norm * beta_k.square()).clamp_min(eps)
    measurement_distance_bound = (
        torch.linalg.vector_norm(weighted_rho, dim=1)
        + beta_k * weighted_H_norm
    ).square().clamp_min(eps)

    measurement_gradient_at_prior = torch.bmm(
        H.transpose(1, 2),
        (R_inv * rho_k).unsqueeze(-1),
    ).squeeze(-1)
    measurement_information_norm = torch.linalg.matrix_norm(HtRinvH, ord=2)
    prior_gradient_bound = beta_k * prior_norm
    measurement_gradient_bound = (
        torch.linalg.vector_norm(measurement_gradient_at_prior, dim=1)
        + beta_k * measurement_information_norm
    )
    geometry_bound = (
        prior_gradient_bound.square() + measurement_gradient_bound.square()
    ).clamp_min(eps)

    kernel_c = (1.0 - spec.q) / (3.0 * spec.q - 1.0)
    kernel_exponent = (2.0 - spec.q) / (1.0 - spec.q)
    contraction_constant = 2.0 * (2.0 - spec.q) / (3.0 * spec.q - 1.0)
    log_kernel_c = math.log(kernel_c)
    log_prior_kernel = -kernel_exponent * F.softplus(
        log_kernel_c
        + torch.log(prior_distance_bound).unsqueeze(1)
        - 2.0 * log_sigma
    )
    log_measurement_kernel = -kernel_exponent * F.softplus(
        log_kernel_c
        + torch.log(measurement_distance_bound).unsqueeze(1)
        - 2.0 * log_sigma
    )

    # With a_i proportional to alpha_i*sigma_i^2, the common normalization
    # cancels between S_k,sigma and the lower-bounded information matrix.
    log_lower_lambda = torch.logsumexp(log_alpha + log_prior_kernel, dim=1)
    log_lower_delta = torch.logsumexp(log_alpha + log_measurement_kernel, dim=1)
    log_bandwidth_sum = torch.logsumexp(log_alpha - 2.0 * log_sigma, dim=1)

    log_scale = torch.maximum(log_lower_lambda, log_lower_delta)
    relative_information = (
        torch.exp(log_lower_lambda - log_scale)[:, None, None] * P_inv
        + torch.exp(log_lower_delta - log_scale)[:, None, None] * HtRinvH
    )
    relative_information = 0.5 * (
        relative_information + relative_information.transpose(1, 2)
    )
    spectral_upper = relative_information.abs().sum(dim=2).amax(dim=1).detach() + 1.0
    relative_information = relative_information + torch.diag_embed(
        (1.0 - mask_sta) * spectral_upper.unsqueeze(1)
    )
    minimum_eigenvalue = torch.linalg.eigvalsh(relative_information)[:, 0].clamp_min(eps)
    log_denominator = log_scale + torch.log(minimum_eigenvalue)
    return (
        math.log(contraction_constant)
        + log_bandwidth_sum
        + torch.log(geometry_bound)
        - log_denominator
        - math.log(spec.contraction_target)
    )


def reconstruction_loss(reconstruction: Tensor, target: Tensor) -> Tensor:
    """L_rec = mean(||rho_hat - rho||_2^2)."""

    if reconstruction.shape != target.shape:
        raise ValueError("reconstruction and physics_target must have identical shapes")
    return (reconstruction - target).square().sum(dim=1).mean()


def dynamic_loss(alpha: Tensor, kernel_scales: Tensor, maha: Tensor) -> Tensor:
    """L_dyn = mean((alpha^T sigma^2) m^2)."""

    metric = _maha_vector(maha, alpha.shape[0])
    weighted_scale = (alpha * kernel_scales.square().unsqueeze(0)).sum(dim=1)
    return (weighted_scale * metric.square()).mean()


def convergence_loss(log_cbar: Tensor, maha: Tensor) -> Tensor:
    """L_conv = mean(softplus(log(C_bar)) m^-2)^2."""

    metric = _maha_vector(maha, log_cbar.shape[0])
    return (F.softplus(log_cbar) * metric.pow(-2)).mean().square()


def training_objective(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    kernel_scales: Tensor,
    mmcf: MmcfSpec,
) -> LossTerms:
    """Evaluate the three self-supervised objectives for one batch."""

    alpha = outputs["alpha"]
    log_cbar = compute_log_cbar(alpha, kernel_scales, batch, mmcf)
    return LossTerms(
        reconstruction=reconstruction_loss(outputs["reconstruction"], batch["physics_target"]),
        dynamic=dynamic_loss(alpha, kernel_scales, batch["maha"]),
        convergence=convergence_loss(log_cbar, batch["maha"]),
    )


def total_loss(terms: LossTerms, weights: LossWeights) -> Tensor:
    """L = w_rec L_rec + w_dyn L_dyn + w_conv L_conv."""

    value = (
        weights.reconstruction * terms.reconstruction
        + weights.dynamic * terms.dynamic
        + weights.convergence * terms.convergence
    )
    if not torch.isfinite(value):
        raise FloatingPointError("total loss is not finite")
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    objective = config["objective"]
    mmcf_spec = MmcfSpec.from_mapping(objective["mmcf"])
    loss_weights = LossWeights.from_mapping(objective["loss_weights"])
    kernel_scales = torch.tensor(objective["kernel_scales"], dtype=torch.float32)
    if kernel_scales.ndim != 1 or kernel_scales.numel() == 0 or torch.any(kernel_scales <= 0.0):
        raise ValueError("kernel_scales must be a non-empty positive vector")

    seed = int(training["seed"])
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    if epochs <= 0 or batch_size <= 1:
        raise ValueError("epochs must be positive and batch_size must exceed one")
    seed_everything(seed)

    dataset = StageDataset(args.data)
    input_dim = int(dataset.arrays["feature"].shape[1])
    target_dim = int(dataset.arrays["physics_target"].shape[1])
    median, scale = fit_normalizer(
        dataset.arrays["feature"],
        min(int(training["scaler_max_samples"]), len(dataset)),
        seed,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SWNMMCF(
        input_dim=input_dim,
        kernel_count=int(kernel_scales.numel()),
        target_dim=target_dim,
        median=median,
        scale=scale,
    ).to(device)
    kernel_scales = kernel_scales.to(device)
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
        totals = {"loss": 0.0, "rec": 0.0, "dyn": 0.0, "conv": 0.0}
        for host_batch in loader:
            batch = {
                name: value.to(device, non_blocking=True)
                for name, value in host_batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["feature"])
            terms = training_objective(outputs, batch, kernel_scales, mmcf_spec)
            loss = total_loss(terms, loss_weights)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["rec"] += float(terms.reconstruction.detach())
            totals["dyn"] += float(terms.dynamic.detach())
            totals["conv"] += float(terms.convergence.detach())

        count = len(loader)
        print(
            f"epoch={epoch + 1:04d} loss={totals['loss'] / count:.6e} "
            f"rec={totals['rec'] / count:.6e} dyn={totals['dyn'] / count:.6e} "
            f"conv={totals['conv'] / count:.6e}"
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": input_dim,
            "kernel_count": int(kernel_scales.numel()),
            "target_dim": target_dim,
        },
        args.checkpoint,
    )

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
        dynamo=False,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--device")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_arguments())
