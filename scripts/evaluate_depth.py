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
from stage1.data.geometry import build_stage1_targets, pixels_to_camera_rays
from stage1.models.model import Stage1CorrespondenceModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Deterministically sample this many evenly spaced manifest entries.",
    )
    return parser.parse_args()


def select_records(dataset: Stage1Dataset, args: argparse.Namespace) -> None:
    start_index = max(int(args.start_index), 0)
    end_index = len(dataset) if args.end_index is None else min(int(args.end_index), len(dataset))
    if start_index >= end_index:
        raise ValueError(f"Invalid manifest range [{start_index}, {end_index}).")
    selected_records = dataset.records[start_index:end_index]
    if args.num_samples is not None:
        num_samples = int(args.num_samples)
        if num_samples <= 0 or num_samples > len(selected_records):
            raise ValueError(
                f"--num_samples must be in [1, {len(selected_records)}], got {num_samples}."
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
    # The camera ray has z=1, so the scalar along the ray is camera-z depth.
    return (delta_xy * ray_xy).sum(dim=-1) / denominator


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = Stage1Dataset(args.manifest)
    select_records(dataset, args)
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

    abs_error_total = 0.0
    squared_error_total = 0.0
    valid_count = 0
    predicted_depth_total = 0.0
    lidar_depth_total = 0.0
    rows = []
    started_at = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (batch_payload, load_errors) in enumerate(loader):
            if batch_payload is None:
                continue
            batch = move_batch_to_device(batch_payload, args.device)
            lidar_depth = batch.lidar_depth.clone()

            # Run the model without LiDAR. The original LiDAR is restored only
            # for the post-hoc reference construction below.
            batch.lidar_depth.zero_()
            batch.lidar_available.zero_()
            output = model(batch)
            batch.lidar_depth = lidar_depth

            targets = build_stage1_targets(
                lidar_depth=lidar_depth,
                query_uv=output.targets.query_uv,
                intrinsics=batch.intrinsics,
                camera_to_world=batch.camera_to_world,
                satellite_meters_per_pixel=batch.satellite_meters_per_pixel,
                satellite_center_world_xy=batch.satellite_center_world_xy,
                satellite_image_hw=(batch.satellite_rgb.shape[-2], batch.satellite_rgb.shape[-1]),
                explicit_negative_mask=batch.explicit_negative_mask,
                lidar_query_radius_px=int(config.get("geometry", {}).get("lidar_query_radius_px", 0)),
            )
            predicted_world_xy = satellite_uv_to_world_xy(
                output.refined_uv,
                batch.satellite_meters_per_pixel,
                batch.satellite_center_world_xy,
                (batch.satellite_rgb.shape[-2], batch.satellite_rgb.shape[-1]),
            )
            predicted_z = world_xy_to_camera_z(
                predicted_world_xy,
                targets.query_uv,
                batch.intrinsics,
                batch.camera_to_world,
            )
            valid = targets.valid_gt_mask & torch.isfinite(predicted_z) & torch.isfinite(targets.lidar_z)
            if valid.any():
                error = predicted_z[valid] - targets.lidar_z[valid]
                abs_error_total += float(error.abs().sum().item())
                squared_error_total += float(error.square().sum().item())
                predicted_depth_total += float(predicted_z[valid].sum().item())
                lidar_depth_total += float(targets.lidar_z[valid].sum().item())
                valid_count += int(error.numel())
            rows.append(
                {
                    "sample_ids": batch.sample_ids,
                    "valid_queries": int(valid.sum().item()),
                    "load_errors": [error.error_message for error in load_errors],
                }
            )
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                elapsed = time.perf_counter() - started_at
                print(
                    f"processed batch {batch_idx + 1}/{len(loader)} "
                    f"({len(batch.sample_ids)} samples, {elapsed:.1f}s)",
                    flush=True,
                )

    summary = {
        "metric_role": "geometric_camera_z_matching_diagnostic",
        "canonical_depth_metric": False,
        "scale_correction_applied": False,
        "primary_depth_evaluator": "evaluate_confidence.py",
        "matching_score_mode": model.matcher.score_mode,
        "matching_score_is_calibrated_probability": False,
        "samples_evaluated": len(dataset),
        "valid_depth_queries": valid_count,
        "depth_mae_m": abs_error_total / max(valid_count, 1),
        "depth_rmse_m": (squared_error_total / max(valid_count, 1)) ** 0.5,
        "mean_predicted_z_m": predicted_depth_total / max(valid_count, 1),
        "mean_lidar_z_m": lidar_depth_total / max(valid_count, 1),
        "lidar_used_for_model_input": False,
    }
    (args.output_dir / "depth_metrics.json").write_text(
        json.dumps({"summary": summary, "batches": rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
