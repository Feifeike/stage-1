from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.dataset_adapter import Stage1ProjectionBatch
from ..data.geometry import pixels_to_camera_rays
from .model import Stage1CorrespondenceModel, Stage1Output
from .refiner import sample_map_at_uv


class ConfidenceHead(nn.Module):
    """Shared per-anchor confidence head used only for scale weighting."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        # Start at exactly equal weights while keeping the branch trainable.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(features).squeeze(-1))


@dataclass
class ConfidenceOutput:
    stage1: Stage1Output
    confidence: torch.Tensor
    front_anchor_features: torch.Tensor
    satellite_anchor_features: torch.Tensor


class Stage1ConfidenceModel(nn.Module):
    """Frozen coarse+refine Stage-1 model plus a trainable confidence head."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.stage1 = Stage1CorrespondenceModel(config)
        base_stage = str(config.get("confidence", {}).get("base_stage", "refine"))
        self.stage1.set_training_stage(base_stage)
        self.stage1.requires_grad_(False)
        self.stage1.eval()

        descriptor_dim = int(config["model"].get("descriptor_dim", 128))
        self.confidence_head = ConfidenceHead(input_dim=2 * descriptor_dim)

    def train(self, mode: bool = True) -> Stage1ConfidenceModel:
        # Keep every original module, including BatchNorm/dropout-like layers,
        # in eval mode even while the new head is trained.
        super().train(mode)
        self.stage1.eval()
        self.confidence_head.train(mode)
        return self

    def forward(self, batch: Stage1ProjectionBatch) -> ConfidenceOutput:
        # LiDAR is never an input to the prediction path. It is retained in the
        # original batch for the separate dense supervision below.
        model_batch = replace(
            batch,
            lidar_depth=torch.zeros_like(batch.lidar_depth),
            lidar_available=torch.zeros_like(batch.lidar_available),
            explicit_negative_mask=torch.zeros_like(batch.explicit_negative_mask),
            negative_available=torch.zeros_like(batch.negative_available),
        )
        with torch.no_grad():
            stage1_output = self.stage1(model_batch)
            front_image_hw = (batch.front_rgb.shape[-2], batch.front_rgb.shape[-1])
            satellite_image_hw = (batch.satellite_rgb.shape[-2], batch.satellite_rgb.shape[-1])
            front_anchor_features = sample_map_at_uv(
                stage1_output.front_features,
                stage1_output.targets.query_uv,
                front_image_hw,
            )
            satellite_anchor_features = sample_map_at_uv(
                stage1_output.satellite_features,
                stage1_output.refined_uv,
                satellite_image_hw,
            )
        features = torch.cat((front_anchor_features, satellite_anchor_features), dim=-1)
        confidence = self.confidence_head(features)
        return ConfidenceOutput(
            stage1=stage1_output,
            confidence=confidence,
            front_anchor_features=front_anchor_features,
            satellite_anchor_features=satellite_anchor_features,
        )


def sample_depth_at_uv(depth: torch.Tensor, uv: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    """Bilinearly sample a `[B,H,W]` camera-z depth map at `[B,N,2]` pixels."""

    image_h, image_w = image_hw
    x = (uv[..., 0] + 0.5) / max(float(image_w), 1.0) * 2.0 - 1.0
    y = (uv[..., 1] + 0.5) / max(float(image_h), 1.0) * 2.0 - 1.0
    grid = torch.stack((x, y), dim=-1).unsqueeze(2)
    sampled = F.grid_sample(
        depth[:, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[:, 0, :, 0]


def resize_depth(depth: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    if tuple(depth.shape[-2:]) == tuple(image_hw):
        return depth.float()
    return F.interpolate(
        depth.float()[:, None], image_hw, mode="bilinear", align_corners=False
    )[:, 0]


def satellite_uv_to_world_xy(
    satellite_uv: torch.Tensor,
    satellite_meters_per_pixel: torch.Tensor,
    satellite_center_world_xy: torch.Tensor,
    satellite_image_hw: tuple[int, int],
) -> torch.Tensor:
    sat_h, sat_w = satellite_image_hw
    center_u = (sat_w - 1) * 0.5
    center_v = (sat_h - 1) * 0.5
    mpp = satellite_meters_per_pixel[:, None].to(satellite_uv.dtype)
    world_x = satellite_center_world_xy[:, None, 0] + (satellite_uv[..., 0] - center_u) * mpp
    world_y = satellite_center_world_xy[:, None, 1] - (satellite_uv[..., 1] - center_v) * mpp
    return torch.stack((world_x, world_y), dim=-1)


def ego_satellite_uv_to_world_xy(
    satellite_uv: torch.Tensor,
    satellite_meters_per_pixel: torch.Tensor,
    satellite_center_world_xy: torch.Tensor,
    satellite_forward_world_xy: torch.Tensor,
    satellite_right_world_xy: torch.Tensor,
    satellite_image_hw: tuple[int, int],
) -> torch.Tensor:
    sat_h, sat_w = satellite_image_hw
    center_u = (sat_w - 1) * 0.5
    center_v = (sat_h - 1) * 0.5
    mpp = satellite_meters_per_pixel[:, None].to(satellite_uv.dtype)
    lateral = (satellite_uv[..., 0] - center_u) * mpp
    forward = (center_v - satellite_uv[..., 1]) * mpp
    return (
        satellite_center_world_xy[:, None, :]
        + lateral[..., None] * satellite_right_world_xy[:, None, :]
        + forward[..., None] * satellite_forward_world_xy[:, None, :]
    )


def refined_world_xy(output: Stage1Output, batch: Stage1ProjectionBatch) -> torch.Tensor:
    satellite_image_hw = (batch.satellite_rgb.shape[-2], batch.satellite_rgb.shape[-1])
    north_up = satellite_uv_to_world_xy(
        output.refined_uv,
        batch.satellite_meters_per_pixel,
        batch.satellite_center_world_xy,
        satellite_image_hw,
    )
    ego = ego_satellite_uv_to_world_xy(
        output.refined_uv,
        batch.satellite_meters_per_pixel,
        batch.satellite_center_world_xy,
        batch.satellite_forward_world_xy,
        batch.satellite_right_world_xy,
        satellite_image_hw,
    )
    use_ego = batch.satellite_is_ego_oriented[:, None, None]
    return torch.where(use_ego, ego, north_up)


def world_xy_to_camera_z(
    world_xy: torch.Tensor,
    query_uv: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
) -> torch.Tensor:
    rays_camera = pixels_to_camera_rays(query_uv, intrinsics)
    rotation = camera_to_world[:, :3, :3]
    origin = camera_to_world[:, :3, 3]
    rays_world = torch.matmul(rays_camera, rotation.transpose(1, 2))
    delta_xy = world_xy - origin[:, None, :2]
    ray_xy = rays_world[..., :2]
    denominator = ray_xy.square().sum(dim=-1).clamp_min(1e-12)
    # The camera ray has z=1, so this scalar is camera-z depth.
    return (delta_xy * ray_xy).sum(dim=-1) / denominator


@dataclass
class ConfidenceDepthInputs:
    anchor_z: torch.Tensor
    mono_anchor_z: torch.Tensor
    anchor_valid: torch.Tensor
    mono_dense: torch.Tensor
    lidar_dense: torch.Tensor
    lidar_valid: torch.Tensor
    dense_valid: torch.Tensor


def build_confidence_depth_inputs(
    output: ConfidenceOutput,
    batch: Stage1ProjectionBatch,
    config: dict,
) -> ConfidenceDepthInputs:
    confidence_cfg = config.get("confidence", {})
    if str(confidence_cfg.get("mono_depth_type", "camera_z")) != "camera_z":
        raise ValueError("confidence.mono_depth_type must be 'camera_z'.")
    front_image_hw = (batch.front_rgb.shape[-2], batch.front_rgb.shape[-1])
    predicted_world_xy = refined_world_xy(output.stage1, batch)
    anchor_z = world_xy_to_camera_z(
        predicted_world_xy,
        output.stage1.targets.query_uv,
        batch.intrinsics,
        batch.camera_to_world,
    )
    mono_image = resize_depth(batch.da3_depth, front_image_hw)
    mono_anchor_z = sample_depth_at_uv(mono_image, output.stage1.targets.query_uv, front_image_hw)
    max_anchor_depth_m = float(confidence_cfg.get("max_anchor_depth_m", 80.0))
    anchor_valid = (
        torch.isfinite(anchor_z)
        & (anchor_z > 0)
        & (anchor_z <= max_anchor_depth_m)
        & torch.isfinite(mono_anchor_z)
        & (mono_anchor_z > 0)
    )

    mono_dense = resize_depth(batch.da3_depth, tuple(batch.lidar_depth.shape[-2:]))
    lidar_dense = batch.lidar_depth.float()
    lidar_min_depth_m = float(confidence_cfg.get("lidar_min_depth_m", 1e-3))
    lidar_max_depth_m = float(confidence_cfg.get("lidar_max_depth_m", 80.0))
    if lidar_min_depth_m < 0 or lidar_max_depth_m <= lidar_min_depth_m:
        raise ValueError("confidence LiDAR depth bounds must satisfy 0 <= min < max.")
    # This is the fixed V used by the existing equal-weight depth protocol.
    # It is independent of confidence and is never used to select anchors.
    lidar_valid = (
        torch.isfinite(lidar_dense)
        & (lidar_dense > lidar_min_depth_m)
        & (lidar_dense < lidar_max_depth_m)
    )
    mono_valid = torch.isfinite(mono_dense) & (mono_dense > 0)
    dense_valid = lidar_valid & mono_valid
    return ConfidenceDepthInputs(
        anchor_z=anchor_z,
        mono_anchor_z=mono_anchor_z,
        anchor_valid=anchor_valid,
        mono_dense=mono_dense,
        lidar_dense=lidar_dense,
        lidar_valid=lidar_valid,
        dense_valid=dense_valid,
    )


def compute_log_scale(
    confidence: torch.Tensor,
    anchor_z: torch.Tensor,
    mono_anchor_z: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    """Compute the differentiable confidence-weighted log scale for one image."""

    if not bool(valid.any().item()):
        return None
    # Sigmoid outputs should already be finite and in (0, 1). Keep the
    # normalization well-defined even if a frozen feature path produces an
    # exceptional value; this is a numerical guard, not a learned gate.
    confidence_valid = torch.nan_to_num(
        confidence[valid].float(), nan=eps, posinf=1.0, neginf=eps
    ).clamp_min(eps)
    log_ratio = torch.log(anchor_z[valid].float().clamp_min(eps)) - torch.log(
        mono_anchor_z[valid].float().clamp_min(eps)
    )
    weights = confidence_valid / confidence_valid.sum().clamp_min(eps)
    log_scale = (weights * log_ratio).sum()
    if not bool(torch.isfinite(log_scale).item()):
        return None
    return log_scale


def compute_equal_log_scale(
    anchor_z: torch.Tensor,
    mono_anchor_z: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    if not bool(valid.any().item()):
        return None
    log_ratio = torch.log(anchor_z[valid].float().clamp_min(eps)) - torch.log(
        mono_anchor_z[valid].float().clamp_min(eps)
    )
    log_scale = log_ratio.mean()
    return log_scale if bool(torch.isfinite(log_scale).item()) else None


def compute_confidence_arithmetic_scale(
    confidence: torch.Tensor,
    anchor_z: torch.Tensor,
    mono_anchor_z: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    """Compute a confidence-weighted arithmetic mean of anchor scale ratios."""

    if not bool(valid.any().item()):
        return None
    confidence_valid = torch.nan_to_num(
        confidence[valid].float(), nan=eps, posinf=1.0, neginf=eps
    ).clamp_min(eps)
    ratios = anchor_z[valid].float() / mono_anchor_z[valid].float().clamp_min(eps)
    finite = torch.isfinite(ratios) & (ratios > 0)
    if not bool(finite.any().item()):
        return None
    ratios = ratios[finite]
    confidence_valid = confidence_valid[finite]
    weights = confidence_valid / confidence_valid.sum().clamp_min(eps)
    scale = (weights * ratios).sum()
    return scale if bool(torch.isfinite(scale).item()) and bool((scale > 0).item()) else None


@dataclass
class ConfidenceLossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    depth_inputs: ConfidenceDepthInputs


def compute_confidence_loss(
    output: ConfidenceOutput,
    batch: Stage1ProjectionBatch,
    config: dict,
) -> ConfidenceLossOutput:
    inputs = build_confidence_depth_inputs(output, batch, config)
    confidence_cfg = config.get("confidence", {})
    eps = float(confidence_cfg.get("depth_epsilon", 1e-6))
    batch_size = output.confidence.shape[0]
    loss_terms: list[torch.Tensor] = []
    weighted_scales: list[torch.Tensor] = []
    equal_scales: list[torch.Tensor] = []
    valid_anchor_total = 0
    lidar_valid_total = 0
    dense_valid_total = 0
    valid_image_count = 0
    no_anchor_count = 0
    no_lidar_count = 0
    no_dense_count = 0
    invalid_scale_count = 0

    for sample_index in range(batch_size):
        anchor_valid = inputs.anchor_valid[sample_index]
        dense_valid = inputs.dense_valid[sample_index]
        valid_anchor_total += int(anchor_valid.sum().item())
        lidar_valid_total += int(inputs.lidar_valid[sample_index].sum().item())
        dense_valid_total += int(dense_valid.sum().item())
        if not bool(anchor_valid.any().item()):
            no_anchor_count += 1
        if not bool(inputs.lidar_valid[sample_index].any().item()):
            no_lidar_count += 1
        if not bool(dense_valid.any().item()):
            no_dense_count += 1

        log_scale = compute_log_scale(
            output.confidence[sample_index],
            inputs.anchor_z[sample_index],
            inputs.mono_anchor_z[sample_index],
            anchor_valid,
            eps,
        )
        equal_log_scale = compute_equal_log_scale(
            inputs.anchor_z[sample_index],
            inputs.mono_anchor_z[sample_index],
            anchor_valid,
            eps,
        )
        if log_scale is None or equal_log_scale is None:
            invalid_scale_count += 1
            continue
        weighted_scales.append(torch.exp(log_scale.clamp(-30.0, 30.0)))
        equal_scales.append(torch.exp(equal_log_scale.clamp(-30.0, 30.0)))
        if not bool(dense_valid.any().item()):
            continue
        log_corrected = torch.log(inputs.mono_dense[sample_index][dense_valid].float().clamp_min(eps)) + log_scale
        log_gt = torch.log(inputs.lidar_dense[sample_index][dense_valid].float().clamp_min(eps))
        loss_terms.append(torch.abs(log_corrected - log_gt).mean())
        valid_image_count += 1

    zero = torch.nan_to_num(output.confidence, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
    loss = torch.stack(loss_terms).mean() if loss_terms else zero
    metric_zero = loss.detach() * 0.0

    def metric(value: float) -> torch.Tensor:
        return loss.detach().new_tensor(value)

    metrics = {
        "loss_confidence": loss.detach(),
        "num_samples": metric(float(batch_size)),
        "num_valid_samples": metric(float(valid_image_count)),
        "num_invalid_samples": metric(float(batch_size - valid_image_count)),
        "num_no_valid_anchors": metric(float(no_anchor_count)),
        "num_no_valid_lidar_pixels": metric(float(no_lidar_count)),
        "num_no_valid_dense_pixels": metric(float(no_dense_count)),
        "num_invalid_scales": metric(float(invalid_scale_count)),
        "num_valid_anchors": metric(float(valid_anchor_total)),
        "num_lidar_valid_pixels": metric(float(lidar_valid_total)),
        "num_loss_valid_pixels": metric(float(dense_valid_total)),
        "confidence_scale_mean": (
            torch.stack(weighted_scales).detach().mean() if weighted_scales else metric_zero
        ),
        "equal_scale_mean": (
            torch.stack(equal_scales).detach().mean() if equal_scales else metric_zero
        ),
    }
    return ConfidenceLossOutput(loss=loss, metrics=metrics, depth_inputs=inputs)
