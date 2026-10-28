from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_KEYS = ("absrel", "sqrel", "rmse", "delta1", "delta2", "delta3")


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def summarize(frames: list[dict], first_summary: dict) -> dict:
    raw = {
        key: mean([float(frame["raw"][key]) for frame in frames])
        for key in METRIC_KEYS
    }
    experiments: dict[str, dict] = {}
    labels = set()
    for frame in frames:
        labels.update(frame.get("experiments", {}).keys())
    for label in sorted(labels, key=lambda value: (not value.isdigit(), value)):
        rows = [frame["experiments"][label] for frame in frames if label in frame.get("experiments", {})]
        methods = set()
        for row in rows:
            methods.update(row.get("methods", {}).keys())
        method_summaries = {}
        for method in sorted(methods):
            method_rows = [row["methods"][method] for row in rows if method in row.get("methods", {})]
            scales = [float(row["scale"]) for row in method_rows if row.get("scale") is not None]
            corrected = {
                key: mean(
                    [float(row["corrected"][key]) for row in method_rows if row.get("corrected") is not None]
                )
                for key in METRIC_KEYS
            }
            status_counts: dict[str, int] = {}
            anchor_errors = []
            for row in method_rows:
                status = str(row.get("status", "unknown"))
                status_counts[status] = status_counts.get(status, 0) + 1
                error = row.get("anchor_error_relative", {}).get("mean")
                if error is not None:
                    anchor_errors.append(float(error))
            method_summary: dict[str, object] = {
                "samples_with_scale": len(scales),
                "samples_without_scale": len(method_rows) - len(scales),
                "status_counts": status_counts,
                "corrected": corrected,
                "scale_mean": mean(scales),
                "scale_min": min(scales) if scales else None,
                "scale_max": max(scales) if scales else None,
                "anchor_relative_error_mean": mean(anchor_errors),
            }
            if method == "ransac":
                method_summary["ransac_inlier_count_mean"] = mean(
                    [float(row.get("final_inlier_count", 0)) for row in method_rows]
                )
                method_summary["ransac_inlier_ratio_mean"] = mean(
                    [float(row.get("inlier_ratio", 0.0)) for row in method_rows]
                )
                inlier_errors = [
                    float(row["final_mean_relative_error"])
                    for row in method_rows
                    if row.get("final_mean_relative_error") is not None
                ]
                method_summary["ransac_inlier_relative_error_mean"] = mean(inlier_errors)
            method_summaries[method] = method_summary

        summary_row: dict[str, object] = {
            "num_anchors": int(rows[0]["num_anchors"]) if rows else None,
            "valid_anchor_count_mean": mean([float(row.get("valid_anchor_count", 0)) for row in rows]),
            "selected_anchor_count_mean": mean([float(row.get("selected_anchor_count", 0)) for row in rows]),
            "methods": method_summaries,
        }
        for key in ("predicted_anchor_depth_range", "right_boundary_inclusive"):
            if rows and key in rows[0]:
                summary_row[key] = rows[0][key]
        experiments[label] = summary_row

    summary = {
        "metric_role": "scale_estimator_ablation",
        "canonical_depth_metric": False,
        "primary_depth_evaluator": "evaluate_confidence.py",
        "samples_requested": len(frames),
        "samples_evaluated": len(frames),
        "valid_lidar_pixels_total": sum(int(frame["valid_lidar_pixels"]) for frame in frames),
        "num_anchors_list": first_summary.get("num_anchors_list"),
        "anchor_depth_ranges": first_summary.get("anchor_depth_ranges"),
        "max_anchor_depth_m": first_summary.get("max_anchor_depth_m"),
        "scale_methods": first_summary.get("scale_methods"),
        "scale_estimators": first_summary.get("scale_estimators"),
        "certainty_source": first_summary.get(
            "certainty_source", first_summary.get("confidence_source")
        ),
        "lidar_used_for_model_input": False,
        "lidar_used_for_scale_estimation": False,
        "raw": raw,
        "experiments": experiments,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.chunks_dir.glob("chunk_*/metrics.json"))
    if not paths:
        raise FileNotFoundError("No chunk metrics found")
    frames = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames.extend(payload["frames"])
    first_summary = json.loads(paths[0].read_text(encoding="utf-8"))["summary"]
    summary = summarize(frames, first_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "frames": frames}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
