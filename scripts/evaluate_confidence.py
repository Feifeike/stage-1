from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from stage1.utils.config import load_config
from stage1.utils.checkpointing import sha256_file
from stage1.models.confidence import (
    Stage1ConfidenceModel,
    build_confidence_depth_inputs,
    compute_confidence_arithmetic_scale,
    compute_equal_log_scale,
    compute_log_scale,
)
from stage1.data.dataset_adapter import Stage1Dataset, build_dataloader, move_batch_to_device
from stage1.evaluation.depth_metrics import compute_depth_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dense depth with confidence-weighted scale correction and report scale ablations."
    )
    parser.add_argument("--config", type=Path, default=Path("models/stage1/configs/config_confidence.yaml"))
    parser.add_argument("--base_checkpoint", type=Path, required=True)
    parser.add_argument("--confidence_checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


def _select_records(dataset: Stage1Dataset, args: argparse.Namespace) -> None:
    start = max(int(args.start_index), 0)
    end = len(dataset) if args.end_index is None else min(int(args.end_index), len(dataset))
    if start >= end:
        raise ValueError(f"Invalid manifest range [{start}, {end}).")
    records = dataset.records[start:end]
    if args.num_samples is not None:
        count = int(args.num_samples)
        if count <= 0 or count > len(records):
            raise ValueError(f"--num_samples must be in [1, {len(records)}].")
        if count == 1:
            indices = [0]
        else:
            indices = [round(index * (len(records) - 1) / (count - 1)) for index in range(count)]
        records = [records[index] for index in indices]
    dataset.records = records


def _load_base(model: Stage1ConfidenceModel, path: Path, device: str) -> str:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    missing, unexpected = model.stage1.load_compatible_checkpoint(state)
    if missing or unexpected:
        print(f"base checkpoint loaded partially: missing={len(missing)}, unexpected={len(unexpected)}")
    return sha256_file(path)


def _load_head(
    model: Stage1ConfidenceModel,
    path: Path,
    device: str,
    base_checkpoint_sha256: str,
) -> str | None:
    payload = torch.load(path, map_location=device, weights_only=False)
    saved_base_hash = payload.get("base_checkpoint_sha256")
    if saved_base_hash is not None and saved_base_hash != base_checkpoint_sha256:
        raise ValueError(
            "The confidence checkpoint was trained with a different Stage-1 base checkpoint: "
            f"saved SHA-256={saved_base_hash}, requested SHA-256={base_checkpoint_sha256}."
        )
    state = payload.get("confidence_head")
    if state is None:
        model_state = payload.get("model", {})
        state = {
            key.removeprefix("confidence_head."): value
            for key, value in model_state.items()
            if key.startswith("confidence_head.")
        }
    model.confidence_head.load_state_dict(state, strict=True)
    return saved_base_hash


def _method_metrics(
    mono: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    log_scale: torch.Tensor,
    eps: float,
) -> dict[str, float]:
    scale = torch.exp(log_scale.clamp(-30.0, 30.0))
    corrected = mono[valid].float() * scale
    target = gt[valid].float()
    finite = torch.isfinite(corrected) & torch.isfinite(target) & (corrected > 0) & (target > 0)
    corrected = corrected[finite]
    target = target[finite]
    return {
        "scale": float(scale.item()),
        **compute_depth_metrics(
            corrected,
            target,
            torch.ones_like(corrected, dtype=torch.bool),
            eps,
        ),
    }


def _summarize(rows: list[dict[str, object]], method: str) -> dict[str, object]:
    valid_rows = [row for row in rows if row[method]["absrel"] is not None]
    method_rows = [row[method] for row in valid_rows]
    anchor_counts = [int(row["valid_anchor_count"]) for row in valid_rows]
    scales = [float(row["scale"]) for row in method_rows]
    pixels = [int(row["valid_pixels"]) for row in method_rows]
    summary = {
        "samples_evaluated": len(method_rows),
        "valid_pixels_total": sum(pixels),
        "valid_anchor_count_total": sum(anchor_counts),
        "valid_anchor_count_mean": sum(anchor_counts) / len(anchor_counts) if anchor_counts else None,
        "valid_anchor_count_min": min(anchor_counts) if anchor_counts else None,
        "valid_anchor_count_max": max(anchor_counts) if anchor_counts else None,
        "scale_mean": sum(scales) / len(scales) if scales else None,
        "scale_min": min(scales) if scales else None,
        "scale_max": max(scales) if scales else None,
    }
    for key in ("absrel", "sqrel", "rmse", "delta1", "delta2", "delta3"):
        values = [float(row[key]) for row in method_rows]
        summary[f"{key}_mean_per_image"] = sum(values) / len(values) if values else None
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = Stage1Dataset(args.manifest)
    _select_records(dataset, args)
    batch_size = args.batch_size or int(config.get("evaluation", {}).get("batch_size", 1))
    num_workers = args.num_workers if args.num_workers is not None else int(config.get("evaluation", {}).get("num_workers", 0))
    loader = build_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = Stage1ConfidenceModel(config).to(args.device)
    base_checkpoint_sha256 = _load_base(model, args.base_checkpoint, args.device)
    confidence_base_sha256 = _load_head(
        model,
        args.confidence_checkpoint,
        args.device,
        base_checkpoint_sha256,
    )
    model.eval()
    eps = float(config.get("confidence", {}).get("depth_epsilon", 1e-6))
    rows: list[dict[str, object]] = []
    invalid_reasons = {"no_valid_anchors": 0, "no_valid_lidar_pixels": 0, "no_valid_dense_pixels": 0}
    with torch.no_grad():
        for batch_index, (batch_payload, load_errors) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            if batch_payload is None:
                continue
            batch = move_batch_to_device(batch_payload, args.device)
            output = model(batch)
            inputs = build_confidence_depth_inputs(output, batch, config)
            for sample_index, sample_id in enumerate(batch.sample_ids):
                anchor_valid = inputs.anchor_valid[sample_index]
                dense_valid = inputs.dense_valid[sample_index]
                if not bool(anchor_valid.any().item()):
                    invalid_reasons["no_valid_anchors"] += 1
                if not bool(inputs.lidar_valid[sample_index].any().item()):
                    invalid_reasons["no_valid_lidar_pixels"] += 1
                if not bool(dense_valid.any().item()):
                    invalid_reasons["no_valid_dense_pixels"] += 1
                weighted_log_scale = compute_log_scale(
                    output.confidence[sample_index],
                    inputs.anchor_z[sample_index],
                    inputs.mono_anchor_z[sample_index],
                    anchor_valid,
                    eps,
                )
                confidence_arithmetic_scale = compute_confidence_arithmetic_scale(
                    output.confidence[sample_index],
                    inputs.anchor_z[sample_index],
                    inputs.mono_anchor_z[sample_index],
                    anchor_valid,
                    eps,
                )
                stage1_certainty = output.stage1.certainty[sample_index]
                certainty_valid = (
                    anchor_valid
                    & torch.isfinite(stage1_certainty)
                    & (stage1_certainty > 0)
                )
                certainty_candidates = torch.nonzero(certainty_valid, as_tuple=False).flatten()
                top20_valid = torch.zeros_like(anchor_valid)
                if certainty_candidates.numel() > 0:
                    certainty_order = torch.argsort(
                        stage1_certainty[certainty_candidates], descending=True
                    )
                    top20_indices = certainty_candidates[certainty_order[:20]]
                    top20_valid[top20_indices] = True
                top20_log_scale = compute_log_scale(
                    stage1_certainty,
                    inputs.anchor_z[sample_index],
                    inputs.mono_anchor_z[sample_index],
                    top20_valid,
                    eps,
                )
                top20_arithmetic_scale = compute_confidence_arithmetic_scale(
                    stage1_certainty,
                    inputs.anchor_z[sample_index],
                    inputs.mono_anchor_z[sample_index],
                    top20_valid,
                    eps,
                )
                equal_log_scale = compute_equal_log_scale(
                    inputs.anchor_z[sample_index],
                    inputs.mono_anchor_z[sample_index],
                    anchor_valid,
                    eps,
                )
                if (
                    weighted_log_scale is None
                    or confidence_arithmetic_scale is None
                    or top20_log_scale is None
                    or top20_arithmetic_scale is None
                    or equal_log_scale is None
                    or not bool(dense_valid.any().item())
                ):
                    empty_metrics = {
                        "scale": None,
                        "absrel": None,
                        "sqrel": None,
                        "rmse": None,
                        "delta1": None,
                        "delta2": None,
                        "delta3": None,
                        "valid_pixels": 0,
                    }
                    rows.append(
                        {
                            "sample_id": str(sample_id),
                            "valid_anchor_count": int(anchor_valid.sum().item()),
                            "valid_lidar_pixels": int(inputs.lidar_valid[sample_index].sum().item()),
                            "valid_dense_pixels": int(dense_valid.sum().item()),
                            "equal_weight": dict(empty_metrics),
                            "confidence_weighted": dict(empty_metrics),
                            "confidence_weighted_arithmetic": dict(empty_metrics),
                            "top20_certainty_geometric": dict(empty_metrics),
                            "top20_certainty_arithmetic": dict(empty_metrics),
                            "load_errors": [error.error_message for error in load_errors],
                        }
                    )
                    continue
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "valid_anchor_count": int(anchor_valid.sum().item()),
                        "valid_lidar_pixels": int(inputs.lidar_valid[sample_index].sum().item()),
                        "valid_dense_pixels": int(dense_valid.sum().item()),
                        "equal_weight": _method_metrics(
                            inputs.mono_dense[sample_index],
                            inputs.lidar_dense[sample_index],
                            dense_valid,
                            equal_log_scale,
                            eps,
                        ),
                        "confidence_weighted": _method_metrics(
                            inputs.mono_dense[sample_index],
                            inputs.lidar_dense[sample_index],
                            dense_valid,
                            weighted_log_scale,
                            eps,
                        ),
                        "confidence_weighted_arithmetic": _method_metrics(
                            inputs.mono_dense[sample_index],
                            inputs.lidar_dense[sample_index],
                            dense_valid,
                            torch.log(confidence_arithmetic_scale.clamp_min(eps)),
                            eps,
                        ),
                        "top20_certainty_geometric": _method_metrics(
                            inputs.mono_dense[sample_index],
                            inputs.lidar_dense[sample_index],
                            dense_valid,
                            top20_log_scale,
                            eps,
                        ),
                        "top20_certainty_arithmetic": _method_metrics(
                            inputs.mono_dense[sample_index],
                            inputs.lidar_dense[sample_index],
                            dense_valid,
                            torch.log(top20_arithmetic_scale.clamp_min(eps)),
                            eps,
                        ),
                        "confidence_mean": float(output.confidence[sample_index][anchor_valid].mean().item()),
                        "load_errors": [error.error_message for error in load_errors],
                    }
                )

    confidence_weighted_summary = _summarize(rows, "confidence_weighted")
    summary = {
        "metric_protocol": "confidence_weighted_scale_corrected_dense_depth_v1",
        "metric_role": "primary_dense_monocular_depth_estimation",
        "primary_depth_metric": "confidence_weighted",
        "primary_scale_method": "learned_confidence_weighted_geometric_mean",
        "primary_depth_metrics": confidence_weighted_summary,
        "scale_weight_source": "Stage1ConfidenceModel.confidence learned sigmoid head; distinct from Stage-1 certainty and matching matchability",
        "scale_formula": "scale = exp(sum_i normalized_confidence_i * log(anchor_z_i / mono_anchor_z_i))",
        "anchor_validity_rule": "all finite positive geometric anchor_z <= max_anchor_depth_m with finite positive sampled mono camera-z; no certainty gate, top-k, or GT filtering",
        "dense_metric_mask_rule": "finite LiDAR camera-z within configured bounds AND finite positive mono depth; fixed independently of confidence and shared by all compared methods",
        "metric_aggregation": "compute each depth metric per image on its valid dense pixels, then report the mean across images",
        "provenance": {
            "config": str(args.config),
            "base_checkpoint": str(args.base_checkpoint),
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "confidence_checkpoint": str(args.confidence_checkpoint),
            "confidence_checkpoint_base_sha256": confidence_base_sha256,
            "manifest": str(args.manifest),
            "device": str(args.device),
            "batch_size": batch_size,
        },
        "samples_requested": len(dataset),
        "samples_processed": len(rows),
        "samples_with_valid_metrics": sum(row["equal_weight"]["absrel"] is not None for row in rows),
        "num_invalid_samples": sum(row["equal_weight"]["absrel"] is None for row in rows),
        "invalid_samples": invalid_reasons,
        "equal_weight": _summarize(rows, "equal_weight"),
        "confidence_weighted": confidence_weighted_summary,
        "confidence_weighted_arithmetic": _summarize(rows, "confidence_weighted_arithmetic"),
        "top20_certainty_geometric": _summarize(rows, "top20_certainty_geometric"),
        "top20_certainty_arithmetic": _summarize(rows, "top20_certainty_arithmetic"),
        "same_lidar_valid_positions": True,
        "candidate_rule": "all finite positive predicted camera-z anchor depths <= max_anchor_depth_m and finite positive sampled mono camera-z depths; no certainty gate, top-k, or GT filtering",
        "lidar_valid_rule": "finite LiDAR camera-z depths with lidar_min_depth_m < z < lidar_max_depth_m; same fixed V for both methods",
        "scale_rule": "scale = exp(sum_i normalized_confidence_i * log(anchor_z_i / mono_anchor_z_i)); weights come from the learned confidence head",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"summary": summary, "frames": rows}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
