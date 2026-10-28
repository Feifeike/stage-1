from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from stage1.utils.config import load_config
from stage1.data.dataset_adapter import Stage1Dataset, build_dataloader, move_batch_to_device
from stage1.training.losses import compute_losses
from stage1.models.model import Stage1CorrespondenceModel
from stage1.scripts.visualize import (
    save_correspondence_visualization,
    save_interactive_correspondence_visualization,
    save_similarity_heatmap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("models/stage1/configs/config.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest to evaluate; overrides the manifest path in the config.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/stage1_eval"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="First manifest index to evaluate (inclusive).",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="Manifest index to stop at (exclusive).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Deterministically sample this many evenly spaced manifest entries.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override evaluation batch size from the config.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override evaluation DataLoader workers from the config.",
    )
    parser.add_argument(
        "--score_mode",
        choices=("forward_softmax", "dual_softmax"),
        default=None,
        help="Override matching.score_mode for this inference/evaluation run.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="K values for coarse recall@K.",
    )
    parser.add_argument(
        "--skip_visualizations",
        action="store_true",
        help="Only compute metrics; do not write the first-batch visualizations.",
    )
    parser.add_argument(
        "--minimal_metrics",
        action="store_true",
        help="Skip the extra diagnostic losses and compute only the requested retrieval/refinement metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.score_mode is not None:
        config.setdefault("matching", {})["score_mode"] = args.score_mode
    manifest = args.manifest or config["data"].get("val_manifest") or config["data"]["train_manifest"]
    dataset = Stage1Dataset(manifest)
    start_index = max(int(args.start_index), 0)
    end_index = len(dataset) if args.end_index is None else min(int(args.end_index), len(dataset))
    if start_index >= end_index:
        raise ValueError(f"Invalid manifest range [{start_index}, {end_index}).")
    selected_records = dataset.records[start_index:end_index]
    if args.num_samples is not None:
        num_samples = int(args.num_samples)
        if num_samples <= 0:
            raise ValueError("--num_samples must be positive.")
        if num_samples > len(selected_records):
            raise ValueError(
                f"--num_samples={num_samples} exceeds the selected manifest range "
                f"of {len(selected_records)} entries."
            )
        if num_samples == 1:
            sample_indices = [0]
        else:
            sample_indices = [
                round(index * (len(selected_records) - 1) / (num_samples - 1))
                for index in range(num_samples)
            ]
        selected_records = [selected_records[index] for index in sample_indices]
    dataset.records = selected_records
    batch_size = args.batch_size or int(config["evaluation"].get("batch_size", 1))
    num_workers = args.num_workers if args.num_workers is not None else int(config["evaluation"].get("num_workers", 0))
    loader = build_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    topk_values = sorted(set(int(k) for k in args.topk if int(k) > 0))
    if not topk_values:
        raise ValueError("--topk must contain at least one positive integer.")
    topk_hits = {k: 0 for k in topk_values}
    topk_denominator = 0
    coarse_distance_total = 0.0
    coarse_pixel_total = 0.0
    coarse_metric_total = 0.0
    coarse_error_count = 0
    refine_pixel_total = 0.0
    refine_metric_total = 0.0
    refine_error_count = 0
    refine_within_5px = 0
    refine_within_10px = 0
    coarse_valid_count = 0
    coarse_invalid_prediction_count = 0
    rows = []
    started_at = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (batch_payload, _) in enumerate(loader):
            if batch_payload is None:
                continue
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch = move_batch_to_device(batch_payload, args.device)
            output = model(batch)
            if args.minimal_metrics:
                record = {}
            else:
                losses = compute_losses(output, config)
                record = {name: float(value.detach().cpu().item()) for name, value in losses.metrics.items()}
            record["sample_ids"] = batch.sample_ids
            rows.append(record)
            for key, value in record.items():
                if isinstance(value, float):
                    totals[key] += value
                    counts[key] += 1
            gt_uv = output.targets.gt_sat_uv
            sat_grid = output.satellite_grid_uv.to(gt_uv.device, gt_uv.dtype)
            grid_dist = torch.cdist(gt_uv.reshape(-1, 1, 2), sat_grid.reshape(1, -1, 2)).squeeze(1)
            gt_indices = grid_dist.argmin(dim=-1).reshape(gt_uv.shape[:2])
            coarse_gt_dist = grid_dist.min(dim=-1).values.reshape(gt_uv.shape[:2])
            coarse_valid = output.targets.valid_gt_mask & (
                coarse_gt_dist <= float(config["loss"].get("coarse_max_gt_grid_dist_px", 32.0))
            )
            num_coarse_valid = int(coarse_valid.sum().item())
            if num_coarse_valid:
                coarse_valid_count += num_coarse_valid
                coarse_distance_total += float(coarse_gt_dist[coarse_valid].sum().item())
                max_k = min(max(topk_values), output.matcher.logits.shape[-1])
                top_indices, _, _ = output.matcher.topk(max_k)
                valid_gt_indices = gt_indices[coarse_valid]
                valid_top_indices = top_indices[coarse_valid]
                for k in topk_values:
                    kk = min(k, max_k)
                    hits = (valid_top_indices[:, :kk] == valid_gt_indices[:, None]).any(dim=1)
                    topk_hits[k] += int(hits.sum().item())
                topk_denominator += num_coarse_valid
            valid_refine = output.targets.valid_gt_mask & output.matcher.coarse_valid_mask
            invalid_coarse_gt_count = int(
                (output.targets.valid_gt_mask & ~output.matcher.coarse_valid_mask).sum().item()
            )
            coarse_invalid_prediction_count += invalid_coarse_gt_count
            if valid_refine.any():
                coarse_error_all = torch.linalg.norm(output.matcher.coarse_uv - gt_uv, dim=-1)
                refine_error_all = torch.linalg.norm(output.refined_uv - gt_uv, dim=-1)
                coarse_error = coarse_error_all[valid_refine]
                refine_error = refine_error_all[valid_refine]
                metric_scale = output.satellite_meters_per_pixel[:, None].expand_as(refine_error_all)[valid_refine]
                coarse_pixel_total += float(coarse_error.sum().item())
                coarse_metric_total += float((coarse_error * metric_scale).sum().item())
                coarse_error_count += int(coarse_error.numel())
                refine_pixel_total += float(refine_error.sum().item())
                refine_metric_total += float((refine_error * metric_scale).sum().item())
                refine_error_count += int(refine_error.numel())
                refine_within_5px += int((refine_error <= 5.0).sum().item())
                refine_within_10px += int((refine_error <= 10.0).sum().item())
            if batch_idx == 0 and not args.skip_visualizations:
                save_correspondence_visualization(batch, output, args.output_dir / "correspondences.png")
                save_similarity_heatmap(batch, output, args.output_dir / "similarity_heatmap.png")
                save_interactive_correspondence_visualization(
                    batch,
                    output,
                    args.output_dir / "correspondences_interactive.html",
                )
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                elapsed = time.perf_counter() - started_at
                print(
                    f"processed batch {batch_idx + 1}/{len(loader)} "
                    f"({len(batch.sample_ids)} samples, {elapsed:.1f}s)",
                    flush=True,
                )
    summary = {key: totals[key] / max(counts[key], 1) for key in sorted(totals)}
    summary["coarse_valid_queries"] = coarse_valid_count
    summary["coarse_gt_grid_distance_global"] = coarse_distance_total / max(coarse_valid_count, 1)
    summary["coarse_error_queries"] = coarse_error_count
    summary["coarse_invalid_predictions"] = coarse_invalid_prediction_count
    summary["coarse_pixel_error_global"] = coarse_pixel_total / max(coarse_error_count, 1)
    summary["coarse_metric_error_global"] = coarse_metric_total / max(coarse_error_count, 1)
    summary["refine_valid_queries"] = refine_error_count
    summary["refine_pixel_error_global"] = refine_pixel_total / max(refine_error_count, 1)
    summary["refine_metric_error_global"] = refine_metric_total / max(refine_error_count, 1)
    summary["refine_within_5px"] = refine_within_5px
    summary["refine_within_5px_fraction"] = refine_within_5px / max(refine_error_count, 1)
    summary["refine_within_10px"] = refine_within_10px
    summary["refine_within_10px_fraction"] = refine_within_10px / max(refine_error_count, 1)
    summary["coarse_topk_denominator"] = topk_denominator
    summary["samples_evaluated"] = len(dataset)
    summary["matching_score_mode"] = model.matcher.score_mode
    summary["matching_score_is_calibrated_probability"] = False
    for k in topk_values:
        summary[f"coarse_recall_at_{k}_hits"] = topk_hits[k]
        summary[f"coarse_recall_at_{k}"] = topk_hits[k] / max(topk_denominator, 1)
    (args.output_dir / "metrics.json").write_text(json.dumps({"summary": summary, "batches": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
