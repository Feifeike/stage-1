from __future__ import annotations

import torch


def compute_depth_metrics(
    predicted_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, float | int | None]:
    """Compute standard positive-depth metrics over a fixed validity mask."""

    valid = (
        valid_mask.bool()
        & torch.isfinite(predicted_depth)
        & (predicted_depth > 0)
        & torch.isfinite(target_depth)
        & (target_depth > 0)
    )
    if not bool(valid.any().item()):
        return {
            "absrel": None,
            "sqrel": None,
            "rmse": None,
            "delta1": None,
            "delta2": None,
            "delta3": None,
            "valid_pixels": 0,
        }

    pred = predicted_depth[valid].float()
    target = target_depth[valid].float()
    error = pred - target
    ratio = torch.maximum(pred / target.clamp_min(eps), target / pred.clamp_min(eps))
    return {
        "absrel": float((error.abs() / target.clamp_min(eps)).mean().item()),
        "sqrel": float((error.square() / target.clamp_min(eps)).mean().item()),
        "rmse": float(error.square().mean().sqrt().item()),
        "delta1": float((ratio < 1.25).float().mean().item()),
        "delta2": float((ratio < (1.25**2)).float().mean().item()),
        "delta3": float((ratio < (1.25**3)).float().mean().item()),
        "valid_pixels": int(target.numel()),
    }
