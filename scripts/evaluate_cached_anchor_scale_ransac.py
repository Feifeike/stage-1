"""Re-evaluate cached sparse refined anchors with weighted and RANSAC scales.

The existing day/night match export already contains the same top-N refined
anchors produced by the Stage-1 checkpoint.  This script avoids another model
forward pass and recomputes only the depth scale and dense LiDAR metrics.
LiDAR participates only in the final dense metrics and is never used to
estimate a scale.
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
from stage1.scripts.evaluate_unidepth_anchor_scale import frame_metrics, sample_depth_at_uv


METRIC_KEYS = ("absrel", "sqrel", "rmse", "delta1", "delta2", "delta3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matches_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_anchors", type=int, default=20)
    parser.add_argument("--max_anchor_depth_m", type=float, default=80.0)
    parser.add_argument("--ransac_rel_threshold", type=float, default=0.15)
    parser.add_argument("--ransac_min_anchors", type=int, default=3)
    parser.add_argument("--ransac_min_inliers", type=int, default=3)
    parser.add_argument("--ransac_min_inlier_ratio", type=float, default=0.5)
    parser.add_argument("--ransac_fallback", choices=("weighted_mean", "none"), default="weighted_mean")
    return parser.parse_args()


def load_depth(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        array = np.load(path).astype(np.float32)
    else:
        array = np.asarray(Image.open(path), dtype=np.float32)
        if array.ndim == 3:
            array = array[..., 0]
        if array.max(initial=0) > 255:
            array = array / 256.0
    return torch.from_numpy(array.astype(np.float32))


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def summarize_anchor_error(anchor_z: torch.Tensor, predicted_z: torch.Tensor, scale: float | None) -> dict:
    if scale is None or anchor_z.numel() == 0:
        return {"mean": None, "median": None, "max": None}
    error = torch.abs(float(scale) * predicted_z - anchor_z) / anchor_z
    return {
        "mean": float(error.mean().item()),
        "median": float(error.median().item()),
        "max": float(error.max().item()),
    }


def main() -> None:
    args = parse_args()
    if args.num_anchors <= 0:
        raise ValueError("num_anchors must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = {str(item["sample_id"]): item for item in manifest}
    match_paths = sorted(args.matches_dir.glob("chunk_*/matches.json"))
    if not match_paths:
        raise FileNotFoundError(f"No chunk_*/matches.json found in {args.matches_dir}")
    matches = []
    for path in match_paths:
        matches.extend(json.loads(path.read_text(encoding="utf-8")))
    match_by_id = {str(row["sampleId"]): row for row in matches}
    missing_matches = [str(item["sample_id"]) for item in manifest if str(item["sample_id"]) not in match_by_id]
    if missing_matches:
        raise ValueError(f"Missing cached matches for {len(missing_matches)} manifest records")

    raw_values = {key: [] for key in METRIC_KEYS}
    method_values = {
        method: {key: [] for key in METRIC_KEYS}
        for method in ("weighted_mean", "ransac")
    }
    scales = {method: [] for method in method_values}
    anchor_errors = {method: [] for method in method_values}
    ransac_inlier_counts: list[float] = []
    ransac_inlier_ratios: list[float] = []
    ransac_inlier_errors: list[float] = []
    status_counts = {method: Counter() for method in method_values}
    frame_rows = []
    total_valid_pixels = 0

    for item in manifest:
        sample_id = str(item["sample_id"])
        cached = match_by_id[sample_id]
        predicted_depth = load_depth(Path(item.get("unidepth_depth_path") or item["depth_path"]))
        lidar_depth = load_depth(Path(item["lidar_depth_path"]))
        if tuple(predicted_depth.shape) != tuple(lidar_depth.shape):
            predicted_depth = F.interpolate(
                predicted_depth[None, None], lidar_depth.shape, mode="bilinear", align_corners=False
            )[0, 0]
        mask = (lidar_depth > 1e-3) & (lidar_depth < 80.0)
        raw = frame_metrics(lidar_depth, predicted_depth, mask)
        if raw is None:
            continue
        for key in METRIC_KEYS:
            raw_values[key].append(raw[key])
        total_valid_pixels += raw["valid_pixels"]

        cached_anchors = cached.get("anchors", [])
        candidate_rows = []
        for anchor in cached_anchors:
            anchor_z = float(anchor.get("anchor_z_m", float("nan")))
            u, v = float(anchor["x"]), float(anchor["y"])
            certainty = float(anchor.get("certainty", anchor.get("confidence", 0.0)))
            if not np.isfinite(anchor_z) or not 0 < anchor_z <= args.max_anchor_depth_m:
                continue
            if not np.isfinite(u) or not np.isfinite(v) or certainty <= 0:
                continue
            if not (0 <= u < predicted_depth.shape[-1] and 0 <= v < predicted_depth.shape[-2]):
                continue
            candidate_rows.append((certainty, int(anchor["id"]), u, v, anchor_z))
        candidate_rows.sort(key=lambda row: -row[0])
        candidate_rows = candidate_rows[: args.num_anchors]
        if candidate_rows:
            uv = torch.tensor([row[2:4] for row in candidate_rows], dtype=torch.float32)[None]
            predicted_anchor_z = sample_depth_at_uv(
                predicted_depth[None], uv, (predicted_depth.shape[-2], predicted_depth.shape[-1])
            )[0]
            anchor_z = torch.tensor([row[4] for row in candidate_rows], dtype=torch.float32)
            certainty = torch.tensor([row[0] for row in candidate_rows], dtype=torch.float32)
            query_ids = [row[1] for row in candidate_rows]
            valid_depth = torch.isfinite(predicted_anchor_z) & (predicted_anchor_z > 0)
            anchor_z = anchor_z[valid_depth]
            predicted_anchor_z = predicted_anchor_z[valid_depth]
            certainty = certainty[valid_depth]
            query_ids = [query_id for query_id, valid in zip(query_ids, valid_depth.tolist()) if valid]
        else:
            anchor_z = torch.empty(0)
            predicted_anchor_z = torch.empty(0)
            certainty = torch.empty(0)
            query_ids = []

        ratios = anchor_z / predicted_anchor_z if anchor_z.numel() else torch.empty(0)
        weighted_mean_value = weighted_mean_scale(ratios.numpy(), certainty.numpy())
        ransac = estimate_ransac_scale(
            anchor_z.numpy(),
            predicted_anchor_z.numpy(),
            weights=certainty.numpy(),
            relative_error_threshold=args.ransac_rel_threshold,
            min_anchors=args.ransac_min_anchors,
            min_inliers=args.ransac_min_inliers,
            min_inlier_ratio=args.ransac_min_inlier_ratio,
            fallback=args.ransac_fallback,
        )
        method_results = {"weighted_mean": weighted_mean_value, "ransac": ransac.scale}
        method_rows = {}
        for method, scale in method_results.items():
            status = "weighted_mean" if method == "weighted_mean" else ransac.status
            status_counts[method][status] += 1
            error_summary = summarize_anchor_error(anchor_z, predicted_anchor_z, scale)
            corrected = frame_metrics(lidar_depth, predicted_depth * float(scale), mask) if scale is not None else None
            method_rows[method] = {
                "scale": scale,
                "status": status,
                "anchor_error_relative": error_summary,
                "corrected": corrected,
            }
            if scale is not None:
                scales[method].append(float(scale))
                anchor_errors[method].append(float(error_summary["mean"]))
                for key in METRIC_KEYS:
                    method_values[method][key].append(corrected[key])
        method_rows["ransac"].update(
            {
                "candidate_scale": ransac.candidate_scale,
                "candidate_inlier_count": ransac.candidate_inlier_count,
                "final_inlier_count": ransac.final_inlier_count,
                "inlier_ratio": ransac.inlier_ratio,
                "candidate_mean_relative_error": ransac.candidate_mean_relative_error,
                "final_mean_relative_error": ransac.final_mean_relative_error,
            }
        )
        ransac_inlier_counts.append(float(ransac.final_inlier_count))
        ransac_inlier_ratios.append(float(ransac.inlier_ratio))
        if ransac.final_mean_relative_error is not None:
            ransac_inlier_errors.append(float(ransac.final_mean_relative_error))

        selected_anchors = []
        for index, query_id in enumerate(query_ids):
            row = {
                "query_index": query_id,
                "anchor_depth_m": float(anchor_z[index].item()),
                "predicted_depth_m": float(predicted_anchor_z[index].item()),
                "scale_i": float(ratios[index].item()),
                "certainty": float(certainty[index].item()),
            }
            for method in method_results:
                scale = method_results[method]
                row[f"{method}_relative_error"] = (
                    float(abs(float(scale) * predicted_anchor_z[index].item() - anchor_z[index].item()) / anchor_z[index].item())
                    if scale is not None
                    else None
                )
            if ransac.inlier_mask:
                row["ransac_inlier"] = bool(ransac.inlier_mask[index])
            selected_anchors.append(row)
        frame_rows.append(
            {
                "sample_id": sample_id,
                "valid_lidar_pixels": raw["valid_pixels"],
                "valid_anchor_count": len(selected_anchors),
                "raw": raw,
                "experiments": {
                    str(args.num_anchors): {
                        "num_anchors": args.num_anchors,
                        "valid_anchor_count": len(selected_anchors),
                        "selected_anchor_count": len(selected_anchors),
                        "selected_anchors": selected_anchors,
                        "methods": method_rows,
                    }
                },
            }
        )

    method_summary = {}
    for method in method_values:
        method_summary[method] = {
            "samples_with_scale": len(scales[method]),
            "samples_without_scale": len(frame_rows) - len(scales[method]),
            "status_counts": dict(status_counts[method]),
            "corrected": {key: mean(method_values[method][key]) for key in METRIC_KEYS},
            "scale_mean": mean(scales[method]),
            "scale_min": min(scales[method]) if scales[method] else None,
            "scale_max": max(scales[method]) if scales[method] else None,
            "anchor_relative_error_mean": mean(anchor_errors[method]),
        }
    method_summary["ransac"].update(
        {
            "ransac_inlier_count_mean": mean(ransac_inlier_counts),
            "ransac_inlier_ratio_mean": mean(ransac_inlier_ratios),
            "ransac_inlier_relative_error_mean": mean(ransac_inlier_errors),
        }
    )
    summary = {
        "metric_role": "scale_estimator_ablation",
        "canonical_depth_metric": False,
        "primary_depth_evaluator": "evaluate_confidence.py",
        "scale_weight_source": "cached Stage-1 certainty; distinct from learned confidence-head weighting",
        "samples_requested": len(manifest),
        "samples_evaluated": len(frame_rows),
        "valid_lidar_pixels_total": total_valid_pixels,
        "num_anchors_list": [args.num_anchors],
        "max_anchor_depth_m": args.max_anchor_depth_m,
        "scale_methods": ["weighted_mean", "ransac"],
        "anchor_source": "cached Stage-1 refined top-N anchors from matches.json",
        "scale_estimators": {
            "weighted_mean": "sum(certainty_i * scale_i) / sum(certainty_i)",
            "ransac": {
                "candidate": "each valid anchor_z / unidepth_z",
                "relative_error": "abs(scale * predicted_depth - anchor_depth) / anchor_depth",
                "relative_error_threshold": args.ransac_rel_threshold,
                "min_anchors": args.ransac_min_anchors,
                "min_inliers": args.ransac_min_inliers,
                "min_inlier_ratio": args.ransac_min_inlier_ratio,
                "fallback": args.ransac_fallback,
                "final_scale": "certainty-weighted mean(scale_i of winning inliers)",
            },
        },
        "certainty_source": "cached Stage-1 certainty (anchor ordering and scale weighting)",
        "lidar_used_for_model_input": False,
        "lidar_used_for_scale_estimation": False,
        "raw": {key: mean(values) for key, values in raw_values.items()},
        "experiments": {
            str(args.num_anchors): {
                "num_anchors": args.num_anchors,
                "valid_anchor_count_mean": mean([frame["valid_anchor_count"] for frame in frame_rows]),
                "selected_anchor_count_mean": mean([frame["valid_anchor_count"] for frame in frame_rows]),
                "methods": method_summary,
            }
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "frames": frame_rows}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
