from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from stage1.utils.config import load_config
from stage1.evaluation.anchor_scale import estimate_ransac_scale, weighted_mean_scale
from stage1.data.dataset_adapter import Stage1Dataset, build_dataloader, move_batch_to_device
from stage1.evaluation.depth_metrics import compute_depth_metrics
from stage1.scripts.evaluate_depth import (
    select_records,
    satellite_uv_to_world_xy,
    world_xy_to_camera_z,
)
from stage1.scripts.evaluate_daynight_matches import ego_satellite_uv_to_world_xy
from stage1.models.model import Stage1CorrespondenceModel
from stage1.models.refiner import pixel_uv_to_normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_anchors", type=int, default=20)
    parser.add_argument(
        "--num_anchors_list",
        type=int,
        nargs="+",
        default=None,
        help="Run several anchor counts from one model forward pass.",
    )
    parser.add_argument(
        "--anchor_depth_ranges",
        type=float,
        nargs=6,
        default=None,
        metavar=("D0", "D1", "D2", "D3", "D4", "D5"),
        help="Run N=5 separately in three predicted-anchor-depth ranges: D0-D1, D2-D3, D4-D5.",
    )
    parser.add_argument(
        "--max_anchor_depth_m",
        type=float,
        default=80.0,
        help="Discard model-derived anchors deeper than this before scale estimation.",
    )
    parser.add_argument(
        "--scale_methods",
        nargs="+",
        choices=("weighted_mean", "ransac"),
        default=["weighted_mean", "ransac"],
        help="Scale estimators to compare. Raw uncorrected depth is always reported.",
    )
    parser.add_argument(
        "--ransac_rel_threshold",
        type=float,
        default=0.15,
        help="RANSAC relative metric-depth residual threshold (default: 15%%).",
    )
    parser.add_argument("--ransac_min_anchors", type=int, default=3)
    parser.add_argument("--ransac_min_inliers", type=int, default=3)
    parser.add_argument("--ransac_min_inlier_ratio", type=float, default=0.5)
    parser.add_argument(
        "--ransac_fallback",
        choices=("weighted_mean", "none"),
        default="weighted_mean",
        help="Fallback when anchors are insufficient or no reliable consensus exists.",
    )
    return parser.parse_args()


def sample_depth_at_uv(depth: torch.Tensor, uv: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    """Bilinearly sample [B,H,W] depth at [B,N,2] pixel coordinates."""

    normalized = pixel_uv_to_normalized(uv, image_hw)
    sampled = F.grid_sample(
        depth[:, None],
        normalized.unsqueeze(2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[:, 0, :, 0]


def frame_metrics(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> dict[str, float] | None:
    metrics = compute_depth_metrics(pred, gt, mask)
    if metrics["valid_pixels"] == 0:
        return None
    return metrics  # type: ignore[return-value]


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _anchor_error_summary(
    anchor_depths: torch.Tensor,
    predicted_depths: torch.Tensor,
    scale: float | None,
) -> dict[str, float | None]:
    if scale is None or anchor_depths.numel() == 0:
        return {"mean": None, "median": None, "max": None}
    errors = torch.abs(float(scale) * predicted_depths - anchor_depths) / anchor_depths
    return {
        "mean": float(errors.mean().item()),
        "median": float(errors.median().item()),
        "max": float(errors.max().item()),
    }


def _corrected_metrics(
    gt: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    scale: float | None,
) -> dict[str, float] | None:
    return frame_metrics(gt, pred * float(scale), mask) if scale is not None else None


def main() -> None:
    args = parse_args()
    if args.max_anchor_depth_m <= 0:
        raise ValueError("max_anchor_depth_m must be positive")
    anchor_counts = sorted(set(int(value) for value in (args.num_anchors_list or [args.num_anchors])))
    if not anchor_counts or any(value <= 0 for value in anchor_counts):
        raise ValueError("Anchor counts must be positive.")
    if args.anchor_depth_ranges is not None:
        depth_values = [float(value) for value in args.anchor_depth_ranges]
        depth_ranges = [(depth_values[index], depth_values[index + 1]) for index in (0, 2, 4)]
        if any(low < 0 or high <= low for low, high in depth_ranges):
            raise ValueError("Anchor depth ranges must have non-negative, increasing bounds.")
    else:
        depth_ranges = []
    if depth_ranges:
        experiment_specs = [
            (f"depth_{int(low)}_{int(high)}", 5, low, high, index == len(depth_ranges) - 1)
            for index, (low, high) in enumerate(depth_ranges)
        ]
    else:
        experiment_specs = [(str(count), count, None, None, False) for count in anchor_counts]

    config = load_config(args.config)
    dataset = Stage1Dataset(args.manifest)
    select_records(dataset, args)
    # KITTI-360 satellite tiles are north-up and have no explicit local
    # orientation in the manifest.  Self-collected tiles carry explicit
    # forward/right axes and are ego-oriented (vehicle heading points up).
    # Keep this distinction instead of using dataset_adapter's synthesized
    # axes for every dataset.
    ego_oriented_by_sample = {
        str(record.sample_id or record.front_rgb_path.stem): (
            record.satellite_forward_world_xy is not None
            and record.satellite_right_world_xy is not None
        )
        for record in dataset.records
    }
    loader = build_dataloader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    model = Stage1CorrespondenceModel(config).to(args.device)
    model.set_training_stage(str(config.get("training", {}).get("stage", "joint")))
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    missing, unexpected = model.load_compatible_checkpoint(payload["model"])
    if missing or unexpected:
        print(
            f"checkpoint loaded partially: missing={len(missing)}, unexpected={len(unexpected)}",
            flush=True,
        )
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = ("absrel", "sqrel", "rmse", "delta1", "delta2", "delta3")
    frame_rows: list[dict[str, object]] = []
    raw_metric_values: dict[str, list[float]] = {key: [] for key in metric_keys}
    corrected_metric_values = {
        label: {method: {key: [] for key in metric_keys} for method in args.scale_methods}
        for label, _, _, _, _ in experiment_specs
    }
    scales_by_experiment = {
        label: {method: [] for method in args.scale_methods}
        for label, _, _, _, _ in experiment_specs
    }
    statuses_by_experiment = {
        label: {method: {} for method in args.scale_methods}
        for label, _, _, _, _ in experiment_specs
    }
    anchor_errors_by_experiment = {
        label: {method: [] for method in args.scale_methods}
        for label, _, _, _, _ in experiment_specs
    }
    ransac_inlier_errors = {label: [] for label, *_ in experiment_specs}
    ransac_inlier_counts = {label: [] for label, *_ in experiment_specs}
    ransac_inlier_ratios = {label: [] for label, *_ in experiment_specs}
    valid_anchor_counts = {label: [] for label, *_ in experiment_specs}
    selected_anchor_counts = {label: [] for label, *_ in experiment_specs}
    total_valid_pixels = 0
    started_at = time.perf_counter()

    def record_status(label: str, method: str, status: str) -> None:
        counts = statuses_by_experiment[label][method]
        counts[status] = int(counts.get(status, 0)) + 1

    with torch.no_grad():
        for batch_idx, (batch_payload, load_errors) in enumerate(loader):
            if batch_payload is None:
                continue
            batch = move_batch_to_device(batch_payload, args.device)
            lidar_depth = batch.lidar_depth.clone()
            # The model sees no LiDAR. This clone is used only for final dense
            # evaluation after the model forward pass.
            batch.lidar_depth.zero_()
            batch.lidar_available.zero_()
            output = model(batch)

            image_hw = (batch.front_rgb.shape[-2], batch.front_rgb.shape[-1])
            uni_depth = batch.da3_depth.float()
            if tuple(uni_depth.shape[-2:]) != image_hw:
                uni_depth_for_anchors = F.interpolate(
                    uni_depth[:, None], image_hw, mode="bilinear", align_corners=False
                )[:, 0]
            else:
                uni_depth_for_anchors = uni_depth
            uni_anchor_depth = sample_depth_at_uv(uni_depth_for_anchors, output.targets.query_uv, image_hw)

            predicted_world_xy_north_up = satellite_uv_to_world_xy(
                output.refined_uv,
                batch.satellite_meters_per_pixel,
                batch.satellite_center_world_xy,
                (batch.satellite_rgb.shape[-2], batch.satellite_rgb.shape[-1]),
            )
            predicted_world_xy_ego = ego_satellite_uv_to_world_xy(
                output.refined_uv,
                batch.satellite_meters_per_pixel,
                batch.satellite_center_world_xy,
                batch.satellite_forward_world_xy,
                batch.satellite_right_world_xy,
                (batch.satellite_rgb.shape[-2], batch.satellite_rgb.shape[-1]),
            )
            ego_oriented = torch.tensor(
                [ego_oriented_by_sample[str(sample_id)] for sample_id in batch.sample_ids],
                dtype=torch.bool,
                device=output.refined_uv.device,
            )[:, None, None]
            predicted_world_xy = torch.where(
                ego_oriented,
                predicted_world_xy_ego,
                predicted_world_xy_north_up,
            )
            # These anchor depths come from the model's refined satellite
            # positions and camera pose, not from LiDAR.
            anchor_z = world_xy_to_camera_z(
                predicted_world_xy,
                output.targets.query_uv,
                batch.intrinsics,
                batch.camera_to_world,
            )
            certainty = output.certainty
            anchor_valid = (
                torch.isfinite(anchor_z)
                & (anchor_z > 0)
                & (anchor_z <= float(args.max_anchor_depth_m))
                & torch.isfinite(uni_anchor_depth)
                & (uni_anchor_depth > 0)
                & torch.isfinite(certainty)
                & (certainty > 0)
            )

            for sample_index, sample_id in enumerate(batch.sample_ids):
                gt = lidar_depth[sample_index]
                pred_raw = uni_depth[sample_index]
                if tuple(pred_raw.shape) != tuple(gt.shape):
                    pred_raw = F.interpolate(
                        pred_raw[None, None], gt.shape, mode="bilinear", align_corners=False
                    )[0, 0]
                mask = (gt > 1e-3) & (gt < 80.0)
                raw = frame_metrics(gt, pred_raw, mask)
                if raw is None:
                    continue

                candidates = torch.nonzero(anchor_valid[sample_index], as_tuple=False).flatten()
                if candidates.numel() > 0:
                    order = torch.argsort(certainty[sample_index, candidates], descending=True)
                    ordered_candidates = candidates[order]
                    ordered_certainty = certainty[sample_index, ordered_candidates]
                    ordered_anchor_z = anchor_z[sample_index, ordered_candidates]
                    ordered_predicted_depth = uni_anchor_depth[sample_index, ordered_candidates]
                    ordered_ratios = ordered_anchor_z / ordered_predicted_depth
                    finite = torch.isfinite(ordered_ratios) & (ordered_ratios > 0)
                    ordered_candidates = ordered_candidates[finite]
                    ordered_certainty = ordered_certainty[finite]
                    ordered_anchor_z = ordered_anchor_z[finite]
                    ordered_predicted_depth = ordered_predicted_depth[finite]
                    ordered_ratios = ordered_ratios[finite]
                else:
                    ordered_candidates = candidates
                    ordered_certainty = torch.empty(0, device=anchor_z.device)
                    ordered_anchor_z = torch.empty(0, device=anchor_z.device)
                    ordered_predicted_depth = torch.empty(0, device=anchor_z.device)
                    ordered_ratios = torch.empty(0, device=anchor_z.device)

                experiments: dict[str, dict[str, object]] = {}
                for label, count, depth_low, depth_high, right_closed in experiment_specs:
                    if depth_low is None:
                        in_range = torch.ones(ordered_anchor_z.shape, dtype=torch.bool, device=anchor_z.device)
                    elif right_closed:
                        in_range = (ordered_anchor_z >= depth_low) & (ordered_anchor_z <= depth_high)
                    else:
                        in_range = (ordered_anchor_z >= depth_low) & (ordered_anchor_z < depth_high)
                    range_indices = torch.nonzero(in_range, as_tuple=False).flatten()
                    valid_count = int(range_indices.numel())
                    selected_indices = range_indices[: min(count, valid_count)]
                    selected_count = int(selected_indices.numel())
                    valid_anchor_counts[label].append(valid_count)
                    selected_anchor_counts[label].append(selected_count)

                    selected_certainty = ordered_certainty[selected_indices]
                    selected_anchor_z = ordered_anchor_z[selected_indices]
                    selected_predicted_depth = ordered_predicted_depth[selected_indices]
                    selected_ratios = ordered_ratios[selected_indices]
                    selected_query_indices = ordered_candidates[selected_indices]

                    weighted_mean = weighted_mean_scale(
                        selected_ratios.detach().cpu().numpy(),
                        selected_certainty.detach().cpu().numpy(),
                    )
                    ransac = estimate_ransac_scale(
                        selected_anchor_z.detach().cpu().numpy(),
                        selected_predicted_depth.detach().cpu().numpy(),
                        weights=selected_certainty.detach().cpu().numpy(),
                        relative_error_threshold=float(args.ransac_rel_threshold),
                        min_anchors=int(args.ransac_min_anchors),
                        min_inliers=int(args.ransac_min_inliers),
                        min_inlier_ratio=float(args.ransac_min_inlier_ratio),
                        fallback=str(args.ransac_fallback),
                    )
                    method_results = {
                        "weighted_mean": (weighted_mean, "weighted_mean", None),
                        "ransac": (
                            ransac.scale,
                            ransac.status,
                            ransac,
                        ),
                    }
                    method_rows: dict[str, dict[str, object]] = {}
                    for method in args.scale_methods:
                        scale, status, ransac_result = method_results[method]
                        record_status(label, method, status)
                        corrected = _corrected_metrics(gt, pred_raw, mask, scale)
                        anchor_error = _anchor_error_summary(
                            selected_anchor_z, selected_predicted_depth, scale
                        )
                        if scale is not None:
                            scales_by_experiment[label][method].append(float(scale))
                            if corrected is not None:
                                for key in metric_keys:
                                    corrected_metric_values[label][method][key].append(corrected[key])
                            if anchor_error["mean"] is not None:
                                anchor_errors_by_experiment[label][method].append(
                                    float(anchor_error["mean"])
                                )
                        method_row: dict[str, object] = {
                            "scale": scale,
                            "status": status,
                            "anchor_error_relative": anchor_error,
                            "corrected": corrected,
                        }
                        if method == "ransac":
                            assert ransac_result is not None
                            ransac_inlier_counts[label].append(ransac_result.final_inlier_count)
                            ransac_inlier_ratios[label].append(ransac_result.inlier_ratio)
                            if ransac_result.final_mean_relative_error is not None:
                                ransac_inlier_errors[label].append(
                                    ransac_result.final_mean_relative_error
                                )
                            method_row.update(
                                {
                                    "candidate_scale": ransac_result.candidate_scale,
                                    "candidate_inlier_count": ransac_result.candidate_inlier_count,
                                    "final_inlier_count": ransac_result.final_inlier_count,
                                    "inlier_ratio": ransac_result.inlier_ratio,
                                    "candidate_mean_relative_error": ransac_result.candidate_mean_relative_error,
                                    "final_mean_relative_error": ransac_result.final_mean_relative_error,
                                }
                            )
                        method_rows[method] = method_row

                    selected_anchors = []
                    for local_index, query_index in enumerate(selected_query_indices):
                        row = {
                            "query_index": int(query_index.item()),
                            "anchor_depth_m": float(selected_anchor_z[local_index].item()),
                            "predicted_depth_m": float(selected_predicted_depth[local_index].item()),
                            "scale_i": float(selected_ratios[local_index].item()),
                            "certainty": float(selected_certainty[local_index].item()),
                        }
                        for method in args.scale_methods:
                            method_scale = method_rows[method]["scale"]
                            row[f"{method}_relative_error"] = (
                                float(
                                    torch.abs(
                                        float(method_scale) * selected_predicted_depth[local_index]
                                        - selected_anchor_z[local_index]
                                    ).div(selected_anchor_z[local_index]).item()
                                )
                                if method_scale is not None
                                else None
                            )
                        if "ransac" in args.scale_methods and ransac.inlier_mask:
                            row["ransac_inlier"] = bool(ransac.inlier_mask[local_index])
                        selected_anchors.append(row)

                    experiments[label] = {
                        "num_anchors": count,
                        "valid_anchor_count": valid_count,
                        "selected_anchor_count": selected_count,
                        "anchor_certainty_min": (
                            float(selected_certainty.min().item()) if selected_count else None
                        ),
                        "anchor_certainty_max": (
                            float(selected_certainty.max().item()) if selected_count else None
                        ),
                        "selected_anchors": selected_anchors,
                        "methods": method_rows,
                    }
                    if depth_low is not None:
                        experiments[label]["predicted_anchor_depth_range"] = [depth_low, depth_high]
                        experiments[label]["right_boundary_inclusive"] = right_closed

                total_valid_pixels += raw["valid_pixels"]
                for key in raw_metric_values:
                    raw_metric_values[key].append(raw[key])
                frame_rows.append(
                    {
                        "sample_id": sample_id,
                        "valid_lidar_pixels": raw["valid_pixels"],
                        "valid_anchor_count": int(ordered_ratios.numel()),
                        "raw": raw,
                        "experiments": experiments,
                    }
                )
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                elapsed = time.perf_counter() - started_at
                print(
                    f"processed batch {batch_idx + 1}/{len(loader)} "
                    f"({len(batch.sample_ids)} samples, {elapsed:.1f}s)",
                    flush=True,
                )

    experiments_summary: dict[str, dict[str, object]] = {}
    for label, count, depth_low, depth_high, right_closed in experiment_specs:
        methods_summary: dict[str, dict[str, object]] = {}
        for method in args.scale_methods:
            scales = scales_by_experiment[label][method]
            method_summary: dict[str, object] = {
                "samples_with_scale": len(scales),
                "samples_without_scale": len(frame_rows) - len(scales),
                "status_counts": statuses_by_experiment[label][method],
                "corrected": {
                    key: _mean(corrected_metric_values[label][method][key])
                    for key in metric_keys
                },
                "scale_mean": _mean(scales),
                "scale_min": min(scales) if scales else None,
                "scale_max": max(scales) if scales else None,
                        "anchor_relative_error_mean": _mean(anchor_errors_by_experiment[label][method]),
            }
            if method == "ransac":
                method_summary.update(
                    {
                        "ransac_inlier_count_mean": _mean(ransac_inlier_counts[label]),
                        "ransac_inlier_ratio_mean": _mean(ransac_inlier_ratios[label]),
                        "ransac_inlier_relative_error_mean": _mean(ransac_inlier_errors[label]),
                    }
                )
            methods_summary[method] = method_summary
        summary_row: dict[str, object] = {
            "num_anchors": count,
            "valid_anchor_count_mean": _mean(valid_anchor_counts[label]),
            "selected_anchor_count_mean": _mean(selected_anchor_counts[label]),
            "methods": methods_summary,
        }
        if depth_low is not None:
            summary_row["predicted_anchor_depth_range"] = [depth_low, depth_high]
            summary_row["right_boundary_inclusive"] = right_closed
        experiments_summary[label] = summary_row

    summary = {
        "metric_role": "scale_estimator_ablation",
        "canonical_depth_metric": False,
        "primary_depth_evaluator": "evaluate_confidence.py",
        "scale_weight_source": "Stage-1 certainty; historical scale-estimator ablation, distinct from learned confidence-head weighting",
        "matching_score_mode": model.matcher.score_mode,
        "matching_score_is_calibrated_probability": False,
        "samples_requested": len(dataset),
        "samples_evaluated": len(frame_rows),
        "valid_lidar_pixels_total": total_valid_pixels,
        "num_anchors_list": anchor_counts,
        "anchor_depth_ranges": depth_ranges or None,
        "max_anchor_depth_m": float(args.max_anchor_depth_m),
        "scale_methods": list(args.scale_methods),
        "scale_estimators": {
            "weighted_mean": "sum(certainty_i * scale_i) / sum(certainty_i)",
            "ransac": {
                "candidate": "each valid anchor_z / unidepth_z",
                "relative_error": "abs(scale * predicted_depth - anchor_depth) / anchor_depth",
                "relative_error_threshold": float(args.ransac_rel_threshold),
                "min_anchors": int(args.ransac_min_anchors),
                "min_inliers": int(args.ransac_min_inliers),
                "min_inlier_ratio": float(args.ransac_min_inlier_ratio),
                "fallback": args.ransac_fallback,
                "final_scale": "certainty-weighted mean(scale_i of winning inliers)",
            },
        },
        "certainty_source": "stage1 output.certainty (ordering and scale weighting)",
        "lidar_used_for_model_input": False,
        "lidar_used_for_scale_estimation": False,
        "raw": {key: _mean(values) for key, values in raw_metric_values.items()},
        "experiments": experiments_summary,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "frames": frame_rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
