"""Run weighted-anchor scale and RANSAC threshold ablations from cached anchors.

The input metrics file must contain per-frame ``selected_anchors``.  The
script recomputes scales for the same candidate pool at several thresholds and
uses LiDAR only for the final dense depth metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from stage1.evaluation.anchor_scale import estimate_ransac_scale, weighted_mean_scale
from stage1.scripts.evaluate_unidepth_anchor_scale import frame_metrics


METRIC_KEYS = ("absrel", "sqrel", "rmse", "delta1", "delta2", "delta3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--experiment", default="200")
    parser.add_argument("--max_depth_m", type=float, default=80.0)
    parser.add_argument(
        "--ransac_rel_thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25],
    )
    parser.add_argument("--ransac_min_anchors", type=int, default=3)
    parser.add_argument("--ransac_min_inliers", type=int, default=3)
    parser.add_argument("--ransac_min_inlier_ratio", type=float, default=0.5)
    parser.add_argument("--ransac_fallback", choices=("weighted_mean", "none"), default="weighted_mean")
    return parser.parse_args()


def load_depth(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        array = np.load(path).astype(np.float32)
    else:
        array = np.asarray(Image.open(path), dtype=np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        if array.size and float(array.max()) > 255.0:
            array = array / 256.0
    return torch.from_numpy(array.astype(np.float32))


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def threshold_label(threshold: float) -> str:
    return f"ransac_{int(round(threshold * 100.0))}pct"


def main() -> None:
    args = parse_args()
    thresholds = sorted(set(float(value) for value in args.ransac_rel_thresholds))
    if not thresholds or any(not 0.0 < value for value in thresholds):
        raise ValueError("RANSAC thresholds must be positive")
    if args.max_depth_m <= 0:
        raise ValueError("max_depth_m must be positive")

    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    frames = payload["frames"]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = {str(item["sample_id"]): item for item in manifest}
    method_labels = ["weighted_mean"] + [threshold_label(value) for value in thresholds]
    method_thresholds = {
        threshold_label(value): value for value in thresholds
    }

    metric_values = {method: {key: [] for key in METRIC_KEYS} for method in method_labels}
    scales = {method: [] for method in method_labels}
    anchor_errors = {method: [] for method in method_labels}
    statuses = {method: Counter() for method in method_labels}
    inlier_counts = {method: [] for method in method_thresholds}
    inlier_ratios = {method: [] for method in method_thresholds}
    inlier_errors = {method: [] for method in method_thresholds}
    raw_values = {key: [] for key in METRIC_KEYS}
    output_frames = []
    total_valid_pixels = 0

    for frame in frames:
        sample_id = str(frame["sample_id"])
        item = records.get(sample_id)
        if item is None:
            raise ValueError(f"Sample {sample_id} is missing from the manifest")
        experiment = frame.get("experiments", {}).get(args.experiment)
        if experiment is None:
            raise ValueError(f"Experiment {args.experiment!r} is missing for {sample_id}")

        predicted = load_depth(Path(item.get("unidepth_depth_path") or item["depth_path"]))
        lidar = load_depth(Path(item["lidar_depth_path"]))
        if tuple(predicted.shape) != tuple(lidar.shape):
            predicted = F.interpolate(
                predicted[None, None], size=tuple(lidar.shape), mode="bilinear", align_corners=False
            )[0, 0]
        mask = (lidar > 1e-3) & (lidar < float(args.max_depth_m))
        raw = frame_metrics(lidar, predicted, mask)
        if raw is None:
            continue
        for key in METRIC_KEYS:
            raw_values[key].append(float(raw[key]))
        total_valid_pixels += int(raw["valid_pixels"])

        anchors = experiment.get("selected_anchors", [])
        anchor_z = np.asarray([float(row["anchor_depth_m"]) for row in anchors], dtype=np.float64)
        predicted_z = np.asarray([float(row["predicted_depth_m"]) for row in anchors], dtype=np.float64)
        certainty = np.asarray(
            [float(row.get("certainty", row.get("confidence", 1.0))) for row in anchors],
            dtype=np.float64,
        )
        valid = (
            np.isfinite(anchor_z)
            & (anchor_z > 0.0)
            & (anchor_z <= float(args.max_depth_m))
            & np.isfinite(predicted_z)
            & (predicted_z > 0.0)
            & np.isfinite(certainty)
            & (certainty > 0.0)
        )
        anchor_z = anchor_z[valid]
        predicted_z = predicted_z[valid]
        certainty = certainty[valid]
        anchor_rows = [row for row, keep in zip(anchors, valid.tolist()) if keep]

        weighted_scale = weighted_mean_scale(anchor_z / predicted_z, certainty)
        ransac_results = {}
        for label, threshold in method_thresholds.items():
            ransac_results[label] = estimate_ransac_scale(
                anchor_z,
                predicted_z,
                weights=certainty,
                relative_error_threshold=threshold,
                min_anchors=args.ransac_min_anchors,
                min_inliers=args.ransac_min_inliers,
                min_inlier_ratio=args.ransac_min_inlier_ratio,
                fallback=args.ransac_fallback,
            )

        method_scales = {"weighted_mean": weighted_scale}
        method_scales.update({label: result.scale for label, result in ransac_results.items()})
        method_rows = {}
        for method, scale in method_scales.items():
            status = "weighted_mean" if method == "weighted_mean" else ransac_results[method].status
            statuses[method][status] += 1
            if scale is None or not anchor_z.size:
                anchor_error = {"mean": None, "max": None}
                corrected = None
            else:
                error = np.abs(float(scale) * predicted_z - anchor_z) / anchor_z
                anchor_error = {"mean": float(error.mean()), "max": float(error.max())}
                corrected = frame_metrics(lidar, predicted * float(scale), mask)
            method_rows[method] = {
                "scale": scale,
                "status": status,
                "anchor_error_relative": anchor_error,
                "corrected": corrected,
            }
            if scale is not None:
                scales[method].append(float(scale))
                if anchor_error["mean"] is not None:
                    anchor_errors[method].append(float(anchor_error["mean"]))
                if corrected is not None:
                    for key in METRIC_KEYS:
                        metric_values[method][key].append(float(corrected[key]))

        for label, result in ransac_results.items():
            method_rows[label].update(
                {
                    "candidate_scale": result.candidate_scale,
                    "candidate_inlier_count": result.candidate_inlier_count,
                    "final_inlier_count": result.final_inlier_count,
                    "inlier_ratio": result.inlier_ratio,
                    "candidate_mean_relative_error": result.candidate_mean_relative_error,
                    "final_mean_relative_error": result.final_mean_relative_error,
                }
            )
            inlier_counts[label].append(float(result.final_inlier_count))
            inlier_ratios[label].append(float(result.inlier_ratio))
            if result.final_mean_relative_error is not None:
                inlier_errors[label].append(float(result.final_mean_relative_error))

        selected_output = []
        ratios = anchor_z / predicted_z
        for index, row in enumerate(anchor_rows):
            output_row = {
                "query_index": row.get("query_index"),
                "anchor_depth_m": float(anchor_z[index]),
                "predicted_depth_m": float(predicted_z[index]),
                "scale_i": float(ratios[index]),
                "certainty": float(certainty[index]),
            }
            for method, scale in method_scales.items():
                output_row[f"{method}_relative_error"] = (
                    float(abs(float(scale) * predicted_z[index] - anchor_z[index]) / anchor_z[index])
                    if scale is not None
                    else None
                )
            for label, result in ransac_results.items():
                output_row[f"{label}_inlier"] = (
                    bool(result.inlier_mask[index]) if result.inlier_mask else None
                )
            selected_output.append(output_row)

        output_frames.append(
            {
                "sample_id": sample_id,
                "valid_lidar_pixels": int(raw["valid_pixels"]),
                "valid_anchor_count": int(experiment.get("valid_anchor_count", len(anchor_rows))),
                "raw": raw,
                "experiments": {
                    args.experiment: {
                        "num_anchors": int(experiment.get("num_anchors", len(selected_output))),
                        "valid_anchor_count": int(experiment.get("valid_anchor_count", len(anchor_rows))),
                        "selected_anchor_count": len(selected_output),
                        "selected_anchors": selected_output,
                        "methods": method_rows,
                    }
                },
            }
        )

    method_summary = {}
    for method in method_labels:
        method_summary[method] = {
            "samples_with_scale": len(scales[method]),
            "samples_without_scale": len(output_frames) - len(scales[method]),
            "status_counts": dict(statuses[method]),
            "corrected": {key: mean(metric_values[method][key]) for key in METRIC_KEYS},
            "scale_mean": mean(scales[method]),
            "scale_min": min(scales[method]) if scales[method] else None,
            "scale_max": max(scales[method]) if scales[method] else None,
            "anchor_relative_error_mean": mean(anchor_errors[method]),
        }
        if method in method_thresholds:
            method_summary[method].update(
                {
                    "relative_error_threshold": method_thresholds[method],
                    "ransac_inlier_count_mean": mean(inlier_counts[method]),
                    "ransac_inlier_ratio_mean": mean(inlier_ratios[method]),
                    "ransac_inlier_relative_error_mean": mean(inlier_errors[method]),
                }
            )

    summary = {
        "metric_role": "scale_estimator_ablation",
        "canonical_depth_metric": False,
        "primary_depth_evaluator": "evaluate_confidence.py",
        "scale_weight_source": "cached Stage-1 certainty; RANSAC threshold ablation, distinct from learned confidence-head weighting",
        "samples_requested": len(frames),
        "samples_evaluated": len(output_frames),
        "valid_lidar_pixels_total": total_valid_pixels,
        "candidate_pool_size_M": int(args.experiment),
        "max_anchor_depth_m": float(args.max_depth_m),
        "scale_methods": method_labels,
        "ransac_relative_error_thresholds": thresholds,
        "scale_estimators": {
            "weighted_mean": "sum(certainty_i * scale_i) / sum(certainty_i) over M candidates",
            "ransac": {
                "candidate": "each valid anchor_z / unidepth_z in the M-candidate pool",
                "relative_error": "abs(scale * predicted_depth - anchor_depth) / anchor_depth",
                "min_anchors": args.ransac_min_anchors,
                "min_inliers": args.ransac_min_inliers,
                "min_inlier_ratio": args.ransac_min_inlier_ratio,
                "fallback": args.ransac_fallback,
                "final_scale": "certainty-weighted mean(scale_i of winning inliers)",
            },
        },
        "certainty_source": "cached Stage-1 certainty",
        "lidar_used_for_model_input": False,
        "lidar_used_for_scale_estimation": False,
        "raw": {key: mean(raw_values[key]) for key in METRIC_KEYS},
        "experiments": {
            args.experiment: {
                "num_anchors": int(args.experiment),
                "selected_anchor_count_mean": mean(
                    [float(frame["experiments"][args.experiment]["selected_anchor_count"]) for frame in output_frames]
                ),
                "methods": method_summary,
            }
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "frames": output_frames}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
