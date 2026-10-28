from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    metric_paths = sorted(args.chunks_dir.glob("chunk_*/metrics.json"))
    match_paths = sorted(args.chunks_dir.glob("chunk_*/matches.json"))
    if not metric_paths or len(metric_paths) != len(match_paths):
        raise FileNotFoundError("Expected metrics.json and matches.json in every chunk directory")
    summaries = [json.loads(path.read_text()) for path in metric_paths]
    samples = []
    for path in match_paths:
        samples.extend(json.loads(path.read_text()))
    summary = {
        "samples_evaluated": sum(item["samples_evaluated"] for item in summaries),
        "num_anchors": summaries[0]["num_anchors"],
        "max_anchor_depth_m": summaries[0]["max_anchor_depth_m"],
        "selected_anchor_count_mean": sum(len(sample["anchors"]) for sample in samples) / max(len(samples), 1),
        "selected_anchor_count_min": min((len(sample["anchors"]) for sample in samples), default=0),
        "selected_anchor_count_max": max((len(sample["anchors"]) for sample in samples), default=0),
        "coarse_valid_queries": sum(item["coarse_valid_queries"] for item in summaries),
        "coarse_pixel_error_sum": sum(item["coarse_pixel_error_sum"] for item in summaries),
        "coarse_metric_error_sum_m": sum(item["coarse_metric_error_sum_m"] for item in summaries),
        "refine_valid_queries": sum(item["refine_valid_queries"] for item in summaries),
        "refine_pixel_error_sum": sum(item["refine_pixel_error_sum"] for item in summaries),
        "refine_metric_error_sum_m": sum(item["refine_metric_error_sum_m"] for item in summaries),
        "refine_within_5px_count": sum(item["refine_within_5px_count"] for item in summaries),
        "refine_within_10px_count": sum(item["refine_within_10px_count"] for item in summaries),
        "coarse_topk_denominator": sum(item["coarse_topk_denominator"] for item in summaries),
        "lidar_used_for_model_input": False,
        "satellite_orientation": summaries[0].get("satellite_orientation", "unspecified"),
        "num_chunks": len(summaries),
    }
    summary["coarse_pixel_error_global"] = summary["coarse_pixel_error_sum"] / max(summary["coarse_valid_queries"], 1)
    summary["coarse_metric_error_global_m"] = summary["coarse_metric_error_sum_m"] / max(summary["coarse_valid_queries"], 1)
    summary["refine_pixel_error_global"] = summary["refine_pixel_error_sum"] / max(summary["refine_valid_queries"], 1)
    summary["refine_metric_error_global_m"] = summary["refine_metric_error_sum_m"] / max(summary["refine_valid_queries"], 1)
    summary["refine_within_5px_fraction"] = summary["refine_within_5px_count"] / max(summary["refine_valid_queries"], 1)
    summary["refine_within_10px_fraction"] = summary["refine_within_10px_count"] / max(summary["refine_valid_queries"], 1)
    for k in [1, 5, 10, 20]:
        hits = sum(item.get(f"coarse_recall_at_{k}_hits", 0) for item in summaries)
        summary[f"coarse_recall_at_{k}_hits"] = hits
        summary[f"coarse_recall_at_{k}"] = hits / max(summary["coarse_topk_denominator"], 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "matches.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
