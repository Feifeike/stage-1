from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from stage1.utils.config import load_config
from stage1.data.dataset_adapter import Stage1Dataset, build_dataloader, move_batch_to_device
from stage1.scripts.evaluate_depth import world_xy_to_camera_z
from stage1.data.geometry import build_stage1_targets
from stage1.models.model import Stage1CorrespondenceModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--num_anchors", type=int, default=20)
    parser.add_argument("--max_anchor_depth_m", type=float, default=80.0)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10, 20])
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def world_xy_to_ego_satellite_uv(
    world_xy: torch.Tensor,
    satellite_meters_per_pixel: torch.Tensor,
    satellite_center_world_xy: torch.Tensor,
    satellite_forward_world_xy: torch.Tensor,
    satellite_right_world_xy: torch.Tensor,
    satellite_image_hw: tuple[int, int],
) -> torch.Tensor:
    """Map world XY into an ego-oriented satellite crop.

    The vehicle/world center is at the tile center, vehicle forward is image-up,
    and vehicle right is image-right.
    """

    sat_h, sat_w = satellite_image_hw
    center_u = (sat_w - 1) * 0.5
    center_v = (sat_h - 1) * 0.5
    delta = world_xy[..., :2] - satellite_center_world_xy[:, None, :]
    u = center_u + (delta * satellite_right_world_xy[:, None, :]).sum(dim=-1) / satellite_meters_per_pixel[:, None]
    v = center_v - (delta * satellite_forward_world_xy[:, None, :]).sum(dim=-1) / satellite_meters_per_pixel[:, None]
    return torch.stack((u, v), dim=-1)


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
    lateral = (satellite_uv[..., 0] - center_u) * satellite_meters_per_pixel[:, None]
    forward = (center_v - satellite_uv[..., 1]) * satellite_meters_per_pixel[:, None]
    return (
        satellite_center_world_xy[:, None, :]
        + lateral[..., None] * satellite_right_world_xy[:, None, :]
        + forward[..., None] * satellite_forward_world_xy[:, None, :]
    )


def main() -> None:
    args = parse_args()
    if args.num_anchors <= 0 or args.max_anchor_depth_m <= 0:
        raise ValueError("num_anchors and max_anchor_depth_m must be positive")
    topk_values = sorted(set(k for k in args.topk if k > 0))
    config = load_config(args.config)
    dataset = Stage1Dataset(args.manifest)
    start_index = max(args.start_index, 0)
    end_index = len(dataset) if args.end_index is None else min(args.end_index, len(dataset))
    if start_index >= end_index:
        raise ValueError(f"Invalid manifest range [{start_index}, {end_index}).")
    dataset.records = dataset.records[start_index:end_index]
    loader = build_dataloader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = Stage1CorrespondenceModel(config).to(args.device)
    model.set_training_stage(str(config.get("training", {}).get("stage", "joint")))
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    missing, unexpected = model.load_compatible_checkpoint(payload["model"])
    if missing or unexpected:
        print(f"checkpoint loaded partially: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    topk_hits = {k: 0 for k in topk_values}
    coarse_denominator = 0
    coarse_pixel_errors: list[float] = []
    refine_pixel_errors: list[float] = []
    coarse_metric_errors: list[float] = []
    refine_metric_errors: list[float] = []
    within_5 = 0
    within_10 = 0
    coarse_invalid_gt_count = 0
    selected_anchor_counts: list[int] = []
    samples: list[dict[str, object]] = []
    started_at = time.perf_counter()
    processed_samples = 0

    with torch.no_grad():
        for batch_idx, (batch_payload, _) in enumerate(loader):
            if batch_payload is None:
                continue
            batch = move_batch_to_device(batch_payload, args.device)
            lidar_depth = batch.lidar_depth.clone()
            # Inference gets zero LiDAR; the clone is used only to build GT metrics.
            batch.lidar_depth.zero_()
            batch.lidar_available.zero_()
            output = model(batch)
            gt_targets = build_stage1_targets(
                lidar_depth=lidar_depth,
                query_uv=output.targets.query_uv,
                intrinsics=batch.intrinsics,
                camera_to_world=batch.camera_to_world,
                satellite_meters_per_pixel=batch.satellite_meters_per_pixel,
                satellite_center_world_xy=batch.satellite_center_world_xy,
                satellite_image_hw=(batch.satellite_rgb.shape[-2], batch.satellite_rgb.shape[-1]),
                lidar_query_radius_px=int(config.get("geometry", {}).get("lidar_query_radius_px", 0)),
            )
            # These satellite crops are ego-oriented, not north-up: vehicle
            # center is at the tile center and vehicle forward points upward.
            gt_uv = world_xy_to_ego_satellite_uv(
                gt_targets.gt_world_xy,
                batch.satellite_meters_per_pixel,
                batch.satellite_center_world_xy,
                batch.satellite_forward_world_xy,
                batch.satellite_right_world_xy,
                (batch.satellite_rgb.shape[-2], batch.satellite_rgb.shape[-1]),
            )
            sat_grid = output.satellite_grid_uv.to(gt_uv.device, gt_uv.dtype)
            grid_dist = torch.cdist(gt_uv.reshape(-1, 1, 2), sat_grid.reshape(1, -1, 2)).squeeze(1)
            gt_indices = grid_dist.argmin(dim=-1).reshape(gt_uv.shape[:2])
            coarse_gt_dist = grid_dist.min(dim=-1).values.reshape(gt_uv.shape[:2])
            coarse_valid = gt_targets.valid_gt_mask & (
                coarse_gt_dist <= float(config["loss"].get("coarse_max_gt_grid_dist_px", 24.0))
            )
            if coarse_valid.any():
                max_k = min(max(topk_values), output.matcher.logits.shape[-1])
                top_indices, _, _ = output.matcher.topk(max_k)
                for k in topk_values:
                    kk = min(k, max_k)
                    topk_hits[k] += int(
                        (top_indices[coarse_valid][:, :kk] == gt_indices[coarse_valid].unsqueeze(1))
                        .any(dim=1)
                        .sum()
                        .item()
                    )
                coarse_denominator += int(coarse_valid.sum().item())

            coarse_invalid_gt_count += int(
                (gt_targets.valid_gt_mask & ~output.matcher.coarse_valid_mask).sum().item()
            )
            valid = gt_targets.valid_gt_mask & output.matcher.coarse_valid_mask
            coarse_error_all = torch.linalg.norm(output.matcher.coarse_uv - gt_uv, dim=-1)
            refine_error_all = torch.linalg.norm(output.refined_uv - gt_uv, dim=-1)
            metric_scale = batch.satellite_meters_per_pixel[:, None].expand_as(refine_error_all)
            if valid.any():
                coarse_pixel_errors.extend(coarse_error_all[valid].cpu().tolist())
                refine_pixel_errors.extend(refine_error_all[valid].cpu().tolist())
                coarse_metric_errors.extend((coarse_error_all[valid] * metric_scale[valid]).cpu().tolist())
                refine_metric_errors.extend((refine_error_all[valid] * metric_scale[valid]).cpu().tolist())
                within_5 += int((refine_error_all[valid] <= 5.0).sum().item())
                within_10 += int((refine_error_all[valid] <= 10.0).sum().item())

            predicted_world_xy = ego_satellite_uv_to_world_xy(
                output.refined_uv,
                batch.satellite_meters_per_pixel,
                batch.satellite_center_world_xy,
                batch.satellite_forward_world_xy,
                batch.satellite_right_world_xy,
                (batch.satellite_rgb.shape[-2], batch.satellite_rgb.shape[-1]),
            )
            anchor_z = world_xy_to_camera_z(
                predicted_world_xy,
                output.targets.query_uv,
                batch.intrinsics,
                batch.camera_to_world,
            )
            anchor_valid = (
                torch.isfinite(anchor_z)
                & (anchor_z > 0)
                & (anchor_z <= args.max_anchor_depth_m)
                & torch.isfinite(output.certainty)
                & (output.certainty > 0)
            )
            for sample_index, sample_id in enumerate(batch.sample_ids):
                candidates = torch.nonzero(anchor_valid[sample_index], as_tuple=False).flatten()
                order = torch.argsort(output.certainty[sample_index, candidates], descending=True) if candidates.numel() else candidates
                selected = candidates[order[: args.num_anchors]]
                selected_anchor_counts.append(int(selected.numel()))
                record = dataset.records[processed_samples]
                points = []
                for query_index in selected.tolist():
                    gt_valid = bool(gt_targets.valid_gt_mask[sample_index, query_index].item())
                    points.append(
                        {
                            "id": int(query_index),
                            "x": float(output.targets.query_uv[sample_index, query_index, 0].item()),
                            "y": float(output.targets.query_uv[sample_index, query_index, 1].item()),
                            "gt_x": float(gt_uv[sample_index, query_index, 0].item()) if gt_valid else None,
                            "gt_y": float(gt_uv[sample_index, query_index, 1].item()) if gt_valid else None,
                            "coarse_x": float(output.matcher.coarse_uv[sample_index, query_index, 0].item()),
                            "coarse_y": float(output.matcher.coarse_uv[sample_index, query_index, 1].item()),
                            "pred_x": float(output.refined_uv[sample_index, query_index, 0].item()),
                            "pred_y": float(output.refined_uv[sample_index, query_index, 1].item()),
                            "certainty": float(output.certainty[sample_index, query_index].item()),
                            "matching_score": float(output.matcher.selected_matching_scores[sample_index, query_index].item()),
                            "anchor_z_m": float(anchor_z[sample_index, query_index].item()),
                            "coarse_error_px": float(coarse_error_all[sample_index, query_index].item()) if gt_valid else None,
                            "refine_error_px": float(refine_error_all[sample_index, query_index].item()) if gt_valid else None,
                            "gt_valid": gt_valid,
                        }
                    )
                samples.append(
                    {
                        "sampleId": str(sample_id),
                        "frontImage": str(record.front_rgb_path),
                        "satelliteImage": str(record.satellite_rgb_path),
                        "frontSize": {"width": int(batch.front_rgb.shape[-1]), "height": int(batch.front_rgb.shape[-2])},
                        "satelliteSize": {"width": int(batch.satellite_rgb.shape[-1]), "height": int(batch.satellite_rgb.shape[-2])},
                        "anchors": points,
                    }
                )
                processed_samples += 1
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                elapsed = time.perf_counter() - started_at
                print(f"processed batch {batch_idx + 1}/{len(loader)} ({len(batch.sample_ids)} samples, {elapsed:.1f}s)", flush=True)

    summary = {
        "matching_score_mode": model.matcher.score_mode if not model.training else "forward_softmax",
        "matching_score_is_calibrated_probability": False,
        "samples_evaluated": len(samples),
        "num_anchors": args.num_anchors,
        "max_anchor_depth_m": args.max_anchor_depth_m,
        "selected_anchor_count_mean": mean([float(v) for v in selected_anchor_counts]),
        "selected_anchor_count_min": min(selected_anchor_counts) if selected_anchor_counts else 0,
        "selected_anchor_count_max": max(selected_anchor_counts) if selected_anchor_counts else 0,
        "coarse_valid_queries": len(coarse_pixel_errors),
        "coarse_invalid_predictions": coarse_invalid_gt_count,
        "coarse_pixel_error_sum": sum(coarse_pixel_errors),
        "coarse_metric_error_sum_m": sum(coarse_metric_errors),
        "coarse_pixel_error_global": mean(coarse_pixel_errors),
        "coarse_metric_error_global_m": mean(coarse_metric_errors),
        "refine_valid_queries": len(refine_pixel_errors),
        "refine_pixel_error_sum": sum(refine_pixel_errors),
        "refine_metric_error_sum_m": sum(refine_metric_errors),
        "refine_pixel_error_global": mean(refine_pixel_errors),
        "refine_metric_error_global_m": mean(refine_metric_errors),
        "refine_within_5px_count": within_5,
        "refine_within_10px_count": within_10,
        "refine_within_5px_fraction": within_5 / max(len(refine_pixel_errors), 1),
        "refine_within_10px_fraction": within_10 / max(len(refine_pixel_errors), 1),
        "coarse_topk_denominator": coarse_denominator,
        "lidar_used_for_model_input": False,
        "satellite_orientation": "ego-oriented: vehicle center at tile center, vehicle forward up",
    }
    for k in topk_values:
        summary[f"coarse_recall_at_{k}_hits"] = topk_hits[k]
        summary[f"coarse_recall_at_{k}"] = topk_hits[k] / max(coarse_denominator, 1)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "matches.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
