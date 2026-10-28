from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1.utils.config import load_config
from stage1.data.dataset_adapter import Stage1Dataset, build_dataloader, move_batch_to_device
from stage1.scripts.evaluate_daynight_matches import ego_satellite_uv_to_world_xy
from stage1.scripts.evaluate_depth import world_xy_to_camera_z
from stage1.models.model import Stage1CorrespondenceModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_anchor_depth_m", type=float, default=80.0)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = Stage1Dataset(args.manifest)
    all_records = dataset.records
    if any(index < 0 or index >= len(all_records) for index in args.indices):
        raise ValueError("An index is outside the manifest range.")
    dataset.records = [all_records[index] for index in args.indices]
    loader = build_dataloader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    model = Stage1CorrespondenceModel(config).to(args.device)
    model.set_training_stage(str(config.get("training", {}).get("stage", "joint")))
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_compatible_checkpoint(payload["model"])
    model.eval()

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch_payload, _load_errors in loader:
            if batch_payload is None:
                continue
            batch = move_batch_to_device(batch_payload, args.device)
            # LiDAR is only an evaluation reference. It must not affect inference.
            batch.lidar_depth.zero_()
            batch.lidar_available.zero_()
            output = model(batch)

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
            certainty = output.certainty
            finite_z = torch.isfinite(anchor_z)
            positive_z = finite_z & (anchor_z > 0)
            depth_ok = positive_z & (anchor_z <= float(args.max_anchor_depth_m))
            finite_certainty = torch.isfinite(certainty)
            certainty_positive = finite_certainty & (certainty > 0)
            valid = depth_ok & certainty_positive

            for sample_index, sample_id in enumerate(batch.sample_ids):
                z = anchor_z[sample_index]
                c = certainty[sample_index]
                z_finite = z[finite_z[sample_index]]
                c_finite = c[finite_certainty[sample_index]]
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "num_queries": int(z.numel()),
                        "finite_z": int(finite_z[sample_index].sum().item()),
                        "z_positive": int(positive_z[sample_index].sum().item()),
                        "z_in_range": int(depth_ok[sample_index].sum().item()),
                        "finite_certainty": int(finite_certainty[sample_index].sum().item()),
                        "certainty_positive": int(certainty_positive[sample_index].sum().item()),
                        "valid_anchor_candidates": int(valid[sample_index].sum().item()),
                        "z_min": float(z_finite.min().item()) if z_finite.numel() else None,
                        "z_max": float(z_finite.max().item()) if z_finite.numel() else None,
                        "certainty_min": float(c_finite.min().item()) if c_finite.numel() else None,
                        "certainty_max": float(c_finite.max().item()) if c_finite.numel() else None,
                    }
                )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
