"""Sparse-anchor scale estimators.

The estimators in this module only use predicted depth and sparse metric anchor
depths.  LiDAR is intentionally not part of this API; it is used by the caller
only when computing the final dense evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RansacScaleResult:
    """Result and diagnostics for one image's scale estimation."""

    scale: float | None
    status: str
    candidate_scale: float | None
    candidate_inlier_count: int
    final_inlier_count: int
    inlier_ratio: float
    candidate_mean_relative_error: float | None
    final_mean_relative_error: float | None
    relative_errors: tuple[float, ...]
    inlier_mask: tuple[bool, ...]


def weighted_mean_scale(scale_values: object, weights: object | None = None) -> float | None:
    """Return a confidence-weighted mean of positive finite scale values."""

    values = np.asarray(scale_values, dtype=np.float64).reshape(-1)
    if weights is None:
        confidence = np.ones_like(values)
    else:
        confidence = np.asarray(weights, dtype=np.float64).reshape(-1)
        if confidence.shape != values.shape:
            raise ValueError("scale_values and weights must have the same shape")
    valid = np.isfinite(values) & (values > 0.0) & np.isfinite(confidence) & (confidence > 0.0)
    values = values[valid]
    confidence = confidence[valid]
    if values.size == 0 or float(confidence.sum()) <= 0.0:
        return None
    return float(np.sum(values * confidence) / np.sum(confidence))


def _empty_result(status: str, scale: float | None = None) -> RansacScaleResult:
    return RansacScaleResult(
        scale=scale,
        status=status,
        candidate_scale=None,
        candidate_inlier_count=0,
        final_inlier_count=0,
        inlier_ratio=0.0,
        candidate_mean_relative_error=None,
        final_mean_relative_error=None,
        relative_errors=(),
        inlier_mask=(),
    )


def estimate_ransac_scale(
    anchor_depths: object,
    predicted_depths: object,
    *,
    relative_error_threshold: float = 0.15,
    min_anchors: int = 3,
    min_inliers: int = 3,
    min_inlier_ratio: float = 0.5,
    weights: object | None = None,
    fallback: str = "weighted_mean",
) -> RansacScaleResult:
    """Estimate one scale from sparse metric anchors using one-point RANSAC.

    Every ``anchor_depth / predicted_depth`` ratio is a candidate.  A candidate
    is scored with the relative metric-depth residual from the request.  The
    candidate with the most inliers wins; ties are broken by its mean inlier
    residual.  The returned RANSAC scale is the confidence-weighted mean ratio
    among the winning inliers.

    ``fallback`` is either ``"weighted_mean"`` (confidence-weighted mean of
    all valid ratios) or ``"none"`` (no correction). Invalid/non-positive
    anchor pairs are discarded before the minimum-anchor check.
    """

    if not 0.0 < float(relative_error_threshold):
        raise ValueError("relative_error_threshold must be positive")
    if int(min_anchors) < 1 or int(min_inliers) < 1:
        raise ValueError("min_anchors and min_inliers must be positive")
    if not 0.0 < float(min_inlier_ratio) <= 1.0:
        raise ValueError("min_inlier_ratio must be in (0, 1]")
    if fallback not in {"weighted_mean", "none"}:
        raise ValueError("fallback must be 'weighted_mean' or 'none'")

    anchors = np.asarray(anchor_depths, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predicted_depths, dtype=np.float64).reshape(-1)
    if anchors.shape != predictions.shape:
        raise ValueError("anchor_depths and predicted_depths must have the same shape")
    if weights is None:
        confidence = np.ones_like(anchors)
    else:
        confidence = np.asarray(weights, dtype=np.float64).reshape(-1)
        if confidence.shape != anchors.shape:
            raise ValueError("anchor_depths and weights must have the same shape")
    valid = (
        np.isfinite(anchors)
        & (anchors > 0.0)
        & np.isfinite(predictions)
        & (predictions > 0.0)
        & np.isfinite(confidence)
        & (confidence > 0.0)
    )
    anchors = anchors[valid]
    predictions = predictions[valid]
    confidence = confidence[valid]
    ratios = anchors / predictions
    fallback_scale = weighted_mean_scale(ratios, confidence)

    def fallback_result(status: str) -> RansacScaleResult:
        if fallback == "weighted_mean" and fallback_scale is not None:
            return _empty_result(status, fallback_scale)
        return _empty_result(status.replace("fallback_weighted_mean", "no_correction"), None)

    if ratios.size < int(min_anchors):
        return fallback_result("fallback_weighted_mean_insufficient_anchors")

    best_index = -1
    best_count = -1
    best_error = float("inf")
    best_errors: np.ndarray | None = None
    best_mask: np.ndarray | None = None
    for index, candidate in enumerate(ratios):
        errors = np.abs(candidate * predictions - anchors) / anchors
        inliers = errors <= float(relative_error_threshold)
        count = int(inliers.sum())
        mean_error = float(errors[inliers].mean()) if count else float("inf")
        if count > best_count or (count == best_count and mean_error < best_error):
            best_index = index
            best_count = count
            best_error = mean_error
            best_errors = errors
            best_mask = inliers

    assert best_index >= 0 and best_errors is not None and best_mask is not None
    inlier_ratio = float(best_count / ratios.size)
    reliable = best_count >= int(min_inliers) and inlier_ratio >= float(min_inlier_ratio)
    if not reliable:
        return fallback_result("fallback_weighted_mean_no_consensus")

    final_scale = weighted_mean_scale(ratios[best_mask], confidence[best_mask])
    assert final_scale is not None
    final_errors = np.abs(final_scale * predictions - anchors) / anchors
    return RansacScaleResult(
        scale=final_scale,
        status="ransac",
        candidate_scale=float(ratios[best_index]),
        candidate_inlier_count=best_count,
        final_inlier_count=best_count,
        inlier_ratio=inlier_ratio,
        candidate_mean_relative_error=best_error,
        final_mean_relative_error=float(final_errors[best_mask].mean()),
        relative_errors=tuple(float(value) for value in final_errors),
        inlier_mask=tuple(bool(value) for value in best_mask),
    )
