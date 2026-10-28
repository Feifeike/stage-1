from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean, median

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1.utils.config import load_config
from stage1.utils.checkpointing import sha256_file
from stage1.models.confidence import ConfidenceHead, ConfidenceOutput, build_confidence_depth_inputs, compute_log_scale
from stage1.scripts.evaluate_confidence import _method_metrics, _summarize
from stage1.models.refiner import sample_map_at_uv
from stage1.data.dataset_adapter import Stage1Dataset, build_dataloader, move_batch_to_device
from stage1.evaluation.depth_metrics import compute_depth_metrics
from stage1.scripts.evaluate_depth import satellite_uv_to_world_xy, world_xy_to_camera_z
from stage1.models.matcher import compute_dual_softmax_log_scores
from stage1.models.model import Stage1CorrespondenceModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare forward, dual-softmax, and mutual-nearest matching on identical logits and queries."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--confidence_checkpoint", type=Path, required=True)
    parser.add_argument("--geometry_threshold_px", type=float, default=10.0)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None, help="Optional smoke run limit.")
    parser.add_argument(
        "--retention_ratios",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="Retention ratios ranked by each mode's matching score; MNN ratios are within accepted mutual pairs.",
    )
    return parser.parse_args()


def _uv_from_indices(
    satellite_grid_uv: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    safe_indices = indices.clamp_min(0)
    uv = satellite_grid_uv.to(device=indices.device)[safe_indices]
    return uv.masked_fill(~valid[..., None], torch.nan)


def _depth_metrics(
    predicted_z: torch.Tensor,
    target_z: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float | int | None]:
    metrics = compute_depth_metrics(predicted_z, target_z, valid_mask)
    return {
        "valid_depth_queries": int(metrics["valid_pixels"] or 0),
        "absrel": metrics["absrel"],
        "sqrel": metrics["sqrel"],
        "rmse_m": metrics["rmse"],
        "delta1": metrics["delta1"],
    }


def _position_summary(errors: list[float], mpp: list[float], valid_gt_count: int) -> dict[str, float | int | None]:
    metric_errors = [error * scale for error, scale in zip(errors, mpp)]
    return {
        "valid_gt_queries": valid_gt_count,
        "position_error_queries": len(errors),
        "invalid_prediction_queries": valid_gt_count - len(errors),
        "mean_error_px": mean(errors) if errors else None,
        "median_error_px": median(errors) if errors else None,
        "mean_error_m": mean(metric_errors) if metric_errors else None,
        "median_error_m": median(metric_errors) if metric_errors else None,
    }


def _retention_summary(
    scores: list[float],
    correct: list[bool],
    ratios: list[float],
    total_valid_gt: int | None = None,
) -> dict[str, dict[str, float | int]]:
    if not scores:
        return {}
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    result: dict[str, dict[str, float | int]] = {}
    for ratio in ratios:
        keep_count = min(len(scores), max(1, int(math.ceil(len(scores) * ratio))))
        kept = order[:keep_count]
        correct_kept = sum(bool(correct[index]) for index in kept)
        key = f"{ratio:g}"
        result[key] = {
            "retained_queries": keep_count,
            "retention_ratio": keep_count / len(scores),
            "precision_at_retained": correct_kept / keep_count,
            "coverage_of_all_valid_gt": correct_kept / max(total_valid_gt or len(scores), 1),
            "ranking_population_queries": len(scores),
        }
    return result


def _mutual_nearest_matches(
    logits: torch.Tensor,
    query_valid_mask: torch.Tensor,
    satellite_valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select row-best pairs only when the satellite also selects that query."""

    pair_valid = (
        query_valid_mask[:, :, None]
        & satellite_valid_mask[:, None, :]
        & torch.isfinite(logits)
    )
    masked_logits = logits.masked_fill(~pair_valid, -torch.inf)
    has_candidate = pair_valid.any(dim=-1)
    safe_logits = torch.where(has_candidate[..., None], masked_logits, torch.zeros_like(masked_logits))
    row_best = safe_logits.argmax(dim=-1).masked_fill(~has_candidate, -1)
    # For every satellite, find its best valid front query within this image.
    has_query = pair_valid.any(dim=-2)
    safe_reverse_logits = torch.where(has_query[:, None, :], masked_logits, torch.zeros_like(masked_logits))
    satellite_best_front = safe_reverse_logits.argmax(dim=-2)
    safe_row_best = row_best.clamp_min(0)
    reverse_for_row_best = satellite_best_front.gather(dim=-1, index=safe_row_best)
    query_indices = torch.arange(logits.shape[1], device=logits.device)[None, :]
    mutual = has_candidate & (reverse_for_row_best == query_indices)
    return row_best.masked_fill(~mutual, -1), mutual


def _score_summary(scores: list[float]) -> dict[str, float | None]:
    return {
        "top_score_mean": mean(scores) if scores else None,
        "top_score_median": median(scores) if scores else None,
        "top_score_min": min(scores) if scores else None,
        "top_score_max": max(scores) if scores else None,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if any(not 0.0 < ratio <= 1.0 for ratio in args.retention_ratios):
        raise ValueError("retention ratios must be in (0, 1]")
    config = load_config(args.config)
    config.setdefault("matching", {})["score_mode"] = "forward_softmax"
    dataset = Stage1Dataset(args.manifest)
    if not dataset.records:
        raise ValueError("The evaluation manifest is empty.")
    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = Stage1CorrespondenceModel(config).to(args.device)
    model.set_training_stage(str(config.get("confidence", {}).get("base_stage", "refine")))
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    missing, unexpected = model.load_compatible_checkpoint(checkpoint["model"])
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    checkpoint_hash = sha256_file(args.checkpoint)
    confidence_hash = sha256_file(args.confidence_checkpoint)
    head_payload = torch.load(args.confidence_checkpoint, map_location=args.device, weights_only=False)
    recorded_hash = head_payload.get("base_checkpoint_sha256")
    if recorded_hash is not None and recorded_hash != checkpoint_hash:
        raise ValueError("Confidence head base checkpoint SHA-256 does not match the requested Stage-1 checkpoint.")
    head = ConfidenceHead(2 * int(config["model"].get("descriptor_dim", 128))).to(args.device)
    head.load_state_dict(head_payload["confidence_head"], strict=True)
    head.eval()
    depth_frames = []
    load_error_records = []
    samples_processed = 0
    eps = float(config.get("confidence", {}).get("depth_epsilon", 1e-6))
    if args.geometry_threshold_px <= 0:
        raise ValueError("geometry_threshold_px must be positive")

    # Store only query-level metrics, not the very large pairwise score matrices.
    names = ("forward_softmax", "dual_softmax", "mnn")
    buckets = {
        name: {
            "coarse_errors_px": [],
            "coarse_scores": [],
            "coarse_correct": [],
            "coarse_exact_grid_correct": [],
            "coarse_mpp": [],
            "refine_errors_px": [],
            "refine_mpp": [],
            "depth_coarse": {"pred": [], "gt": []},
            "depth_refine": {"pred": [], "gt": []},
            "invalid_predictions": 0,
        }
        for name in names
    }
    valid_gt_queries = 0
    query_count = 0
    valid_input_queries = 0
    satellite_candidates = 0
    mpp_values: list[float] = []
    batches_processed = 0
    started = time.perf_counter()
    threshold_px = float(args.geometry_threshold_px)
    grid_eligibility_px = float(config.get("loss", {}).get("coarse_max_gt_grid_dist_px", 24.0))

    with torch.no_grad():
        for batch_idx, (batch_payload, load_errors) in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            load_error_records.extend(str(error.error_message) for error in load_errors)
            if batch_payload is None:
                continue
            batch = move_batch_to_device(batch_payload, args.device)
            output = model(batch)
            samples_processed += len(batch.sample_ids)
            if output.matcher.score_mode != "forward_softmax":
                raise RuntimeError("A/B evaluation must obtain its shared logits in forward_softmax mode.")
            batches_processed += 1
            bsz, n_queries, _ = output.matcher.logits.shape
            query_count += bsz * n_queries
            valid_input_queries += int(output.matcher.query_valid_mask.sum().item())
            satellite_candidates += int(output.matcher.satellite_valid_mask.sum().item())
            mpp_values.extend(float(value) for value in batch.satellite_meters_per_pixel.cpu().tolist())

            # All modes are selected from the same already scaled logits. GT
            # is used only after scoring and never affects pair selection.
            dual_log_scores = compute_dual_softmax_log_scores(
                output.matcher.logits,
                output.matcher.query_valid_mask,
                output.matcher.satellite_valid_mask,
            )
            mnn_indices, mnn_valid = _mutual_nearest_matches(
                output.matcher.logits,
                output.matcher.query_valid_mask,
                output.matcher.satellite_valid_mask,
            )
            mnn_uv = _uv_from_indices(output.satellite_grid_uv, mnn_indices, mnn_valid)
            mnn_forward_score = output.matcher.probabilities.gather(
                -1, mnn_indices.clamp_min(0)[..., None]
            ).squeeze(-1).masked_fill(~mnn_valid, 0.0)
            common_gt = output.targets.valid_gt_mask & output.matcher.query_valid_mask
            valid_gt_queries += int(common_gt.sum().item())
            common_prediction_valid = output.matcher.coarse_valid_mask
            grid_dist = torch.cdist(
                output.targets.gt_sat_uv.reshape(-1, 1, 2),
                output.satellite_grid_uv.to(
                    output.targets.gt_sat_uv.device, output.targets.gt_sat_uv.dtype
                ).reshape(1, -1, 2),
            ).squeeze(1)
            gt_grid_indices = grid_dist.argmin(dim=-1).reshape(common_gt.shape)
            nearest_grid_distance = grid_dist.min(dim=-1).values.reshape(common_gt.shape)
            coarse_gt_eligible = common_gt & (nearest_grid_distance <= grid_eligibility_px)
            modes = {
                "forward_softmax": {
                    "indices": output.matcher.coarse_indices,
                    "uv": output.matcher.coarse_uv,
                    "score": output.matcher.probabilities.max(dim=-1).values,
                    "refined_uv": output.refined_uv,
                },
                "dual_softmax": {
                    "indices": dual_log_scores.argmax(dim=-1).masked_fill(~common_prediction_valid, -1),
                    "uv": None,
                    "score": dual_log_scores.max(dim=-1).values.exp(),
                    "valid": common_prediction_valid,
                    "refined_uv": None,
                },
                "mnn": {
                    "indices": mnn_indices,
                    "uv": mnn_uv,
                    # Rank accepted mutual pairs by their row-forward probability.
                    "score": mnn_forward_score,
                    "valid": mnn_valid,
                    "refined_uv": None,
                },
            }
            for values in modes.values():
                if "valid" not in values:
                    values["valid"] = common_prediction_valid
                if values["uv"] is None:
                    values["uv"] = _uv_from_indices(
                        output.satellite_grid_uv,
                        values["indices"],
                        values["valid"],
                    )
            for name, values in modes.items():
                if values["refined_uv"] is None:
                    refined_uv, _ = model.refine_from_coarse(
                        batch=batch,
                        output_features=(output.front_features, output.satellite_features),
                        front_query_uv=output.targets.query_uv,
                        coarse_uv=values["uv"],
                        match_valid_mask=values["valid"],
                        initial_matchability_logits=output.matcher.matchability_logits,
                    )
                    values["refined_uv"] = refined_uv

            # Evaluate the same fixed head at each mode's selected locations.
            # No GT mask, certainty gate or score threshold selects scale anchors.
            front_hw = tuple(batch.front_rgb.shape[-2:])
            satellite_hw = tuple(batch.satellite_rgb.shape[-2:])
            front_features = sample_map_at_uv(output.front_features, output.targets.query_uv, front_hw)
            batch_depth = {name: {} for name in names}
            for name, values in modes.items():
                for stage, location_key in (("coarse", "uv"), ("refined", "refined_uv")):
                    locations = values[location_key]
                    location_valid = values["valid"] & torch.isfinite(locations).all(dim=-1)
                    safe_locations = torch.where(location_valid[..., None], locations, torch.zeros_like(locations))
                    satellite_features = sample_map_at_uv(output.satellite_features, safe_locations, satellite_hw)
                    confidence = head(torch.cat((front_features, satellite_features), dim=-1))
                    stage_output = replace(output, refined_uv=locations)
                    inputs = build_confidence_depth_inputs(
                        ConfidenceOutput(stage_output, confidence, front_features, satellite_features), batch, config
                    )
                    inputs.anchor_valid &= location_valid
                    stage_rows = []
                    for sample_index in range(bsz):
                        anchor_valid = inputs.anchor_valid[sample_index]
                        valid = inputs.dense_valid[sample_index]
                        log_scale = compute_log_scale(confidence[sample_index], inputs.anchor_z[sample_index],
                                                      inputs.mono_anchor_z[sample_index], anchor_valid, eps)
                        metrics = None
                        if log_scale is not None and bool(valid.any()):
                            metrics = _method_metrics(inputs.mono_dense[sample_index], inputs.lidar_dense[sample_index],
                                                      valid, log_scale, eps)
                        stage_rows.append({"valid_anchor_count": int(anchor_valid.sum()),
                                           "valid_dense_pixels": int(valid.sum()), "metrics": metrics})
                    batch_depth[name][stage] = (stage_rows, inputs.dense_valid)
            for stage in ("coarse", "refined"):
                if not all(torch.equal(batch_depth[names[0]][stage][1], batch_depth[name][stage][1]) for name in names[1:]):
                    raise RuntimeError("Matching-mode dense depth evaluation masks differ")
            for sample_index, sample_id in enumerate(batch.sample_ids):
                depth_frames.append({"sample_id": str(sample_id), "modes": {
                    name: {stage: batch_depth[name][stage][0][sample_index] for stage in ("coarse", "refined")}
                    for name in names}})

            gt_z = output.targets.lidar_z
            gt_position = output.targets.gt_sat_uv
            query_uv = output.targets.query_uv
            for name, values in modes.items():
                pred_uv = values["uv"]
                refined_uv = values["refined_uv"]
                prediction_valid = values["valid"] & torch.isfinite(pred_uv).all(dim=-1)
                gt_scored = common_gt & prediction_valid
                errors = torch.linalg.norm(pred_uv - gt_position, dim=-1)
                buckets[name]["invalid_predictions"] += int((common_gt & ~prediction_valid).sum().item())
                valid_errors = common_gt & prediction_valid & torch.isfinite(errors)
                buckets[name]["coarse_errors_px"].extend(errors[valid_errors].cpu().tolist())
                buckets[name]["coarse_mpp"].extend(
                    batch.satellite_meters_per_pixel[:, None].expand_as(errors)[valid_errors].cpu().tolist()
                )
                scores = values["score"]
                safe_scores = torch.where(
                    prediction_valid & torch.isfinite(scores), scores, torch.zeros_like(scores)
                )
                score_queries = common_gt & prediction_valid
                buckets[name]["coarse_scores"].extend(safe_scores[score_queries].float().cpu().tolist())
                correct = valid_errors & (errors <= threshold_px)
                buckets[name]["coarse_correct"].extend(correct[score_queries].cpu().tolist())
                exact_correct = (values["indices"] == gt_grid_indices) & prediction_valid
                buckets[name]["coarse_exact_grid_correct"].extend(
                    exact_correct[coarse_gt_eligible].cpu().tolist()
                )

                refined_error = torch.linalg.norm(refined_uv - gt_position, dim=-1)
                refined_valid = common_gt & prediction_valid & torch.isfinite(refined_error)
                buckets[name]["refine_errors_px"].extend(refined_error[refined_valid].cpu().tolist())
                buckets[name]["refine_mpp"].extend(
                    batch.satellite_meters_per_pixel[:, None].expand_as(refined_error)[refined_valid].cpu().tolist()
                )

                pred_world_xy_coarse = satellite_uv_to_world_xy(
                    pred_uv,
                    batch.satellite_meters_per_pixel,
                    batch.satellite_center_world_xy,
                    (batch.satellite_rgb.shape[-2], batch.satellite_rgb.shape[-1]),
                )
                pred_world_xy_refine = satellite_uv_to_world_xy(
                    refined_uv,
                    batch.satellite_meters_per_pixel,
                    batch.satellite_center_world_xy,
                    (batch.satellite_rgb.shape[-2], batch.satellite_rgb.shape[-1]),
                )
                pred_z_coarse = world_xy_to_camera_z(
                    pred_world_xy_coarse, query_uv, batch.intrinsics, batch.camera_to_world
                )
                pred_z_refine = world_xy_to_camera_z(
                    pred_world_xy_refine, query_uv, batch.intrinsics, batch.camera_to_world
                )
                for stage_name, pred_z in (("depth_coarse", pred_z_coarse), ("depth_refine", pred_z_refine)):
                    depth_valid = (
                        common_gt
                        & prediction_valid
                        & torch.isfinite(pred_z)
                        & (pred_z > 0)
                        & torch.isfinite(gt_z)
                        & (gt_z > 0)
                    )
                    buckets[name][stage_name]["pred"].extend(pred_z[depth_valid].float().cpu().tolist())
                    buckets[name][stage_name]["gt"].extend(gt_z[depth_valid].float().cpu().tolist())

            print(
                f"processed batch {batch_idx + 1}/{len(loader)} ({len(batch.sample_ids)} samples, "
                f"{time.perf_counter() - started:.1f}s)",
                flush=True,
            )

    if batches_processed == 0:
        raise RuntimeError("No usable batches were evaluated.")
    summaries: dict[str, dict[str, object]] = {}
    for name in names:
        bucket = buckets[name]
        matched_queries = len(bucket["coarse_scores"])
        correct_matches = sum(bucket["coarse_correct"])
        summaries[name] = {
            "top1_geometry_accuracy": sum(bucket["coarse_correct"]) / max(valid_gt_queries, 1),
            "geometry_threshold_px": threshold_px,
            "matched_gt_queries": matched_queries,
            "unmatched_gt_queries": max(valid_gt_queries - matched_queries, 0),
            "matched_query_coverage": matched_queries / max(valid_gt_queries, 1),
            "matched_pair_precision_at_threshold": correct_matches / max(matched_queries, 1),
            "exact_grid_top1_accuracy": (
                sum(bucket["coarse_exact_grid_correct"]) / len(bucket["coarse_exact_grid_correct"])
                if bucket["coarse_exact_grid_correct"]
                else None
            ),
            "exact_grid_top1_queries": len(bucket["coarse_exact_grid_correct"]),
            "coarse_position": _position_summary(
                bucket["coarse_errors_px"], bucket["coarse_mpp"], valid_gt_queries
            ),
            "top_matching_score": _score_summary(bucket["coarse_scores"]),
            "equal_retention": _retention_summary(
                bucket["coarse_scores"],
                bucket["coarse_correct"],
                args.retention_ratios,
                total_valid_gt=valid_gt_queries,
            ),
            "retention_population": (
                "accepted mutual MNN pairs; coverage is measured against all valid GT queries"
                if name == "mnn"
                else "all valid GT queries"
            ),
            "invalid_prediction_queries": bucket["invalid_predictions"],
            "refined_position": _position_summary(
                bucket["refine_errors_px"], bucket["refine_mpp"], valid_gt_queries
            ),
            "depth_coarse": _depth_metrics(
                torch.tensor(bucket["depth_coarse"]["pred"]),
                torch.tensor(bucket["depth_coarse"]["gt"]),
                torch.ones(len(bucket["depth_coarse"]["pred"]), dtype=torch.bool),
            ),
            "depth_refine": _depth_metrics(
                torch.tensor(bucket["depth_refine"]["pred"]),
                torch.tensor(bucket["depth_refine"]["gt"]),
                torch.ones(len(bucket["depth_refine"]["pred"]), dtype=torch.bool),
            ),
        }
    if sha256_file(args.checkpoint) != checkpoint_hash or sha256_file(args.confidence_checkpoint) != confidence_hash:
        raise RuntimeError("Checkpoint files changed during evaluation")
    paired_depth_counts = {}
    for stage in ("coarse", "refined"):
        paired = [frame for frame in depth_frames if all(
            frame["modes"][name][stage]["metrics"] is not None
            and frame["modes"][name][stage]["metrics"]["absrel"] is not None for name in names)]
        paired_depth_counts[stage] = {"paired_frames": len(paired), "excluded_frames": len(depth_frames) - len(paired)}
        for name in names:
            rows = [{"valid_anchor_count": frame["modes"][name][stage]["valid_anchor_count"],
                     "confidence_weighted": frame["modes"][name][stage]["metrics"]} for frame in paired]
            summaries[name].setdefault("confidence_weighted_depth", {})[stage] = _summarize(rows, "confidence_weighted")
    summary = {
        "experiment": "B1 forward, dual-softmax and MNN inference ablation",
        "comparison": "All modes are computed from the same checkpoint, batches, preprocessing, and temperature-scaled logits.",
        "score_definitions": {
            "forward_softmax": "softmax(L, dim=-1)",
            "dual_softmax": "exp(log_softmax(float32(L), dim=-1) + log_softmax(float32(L), dim=-2))",
            "mnn": "accept the row-argmax pair only when the selected satellite's column-argmax is the same front query; no dustbin",
            "mnn_retention_score": "forward-softmax probability of the accepted mutual pair; retention is ranked within accepted MNN pairs",
            "mnn_unmatched": "non-mutual queries receive index -1 and no coarse/refined match; counted as unmatched and as incorrect in all-query Top-1",
            "mnn_refinement": "same dense refiner; unmatched queries are masked in its output and zero-filled as internal coarse-warp context",
            "temperature": "L is already temperature/logit-scale adjusted; no extra scaling is applied.",
            "dual_score_calibration": "raw dual score, not a calibrated match-correctness probability.",
            "invalid_positions": "masked from both normalizations; reverse normalization uses every valid front query in each sample.",
            "training": "original row-wise cross entropy on unchanged logits; no reverse loss.",
        },
        "provenance": {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "manifest": str(args.manifest),
            "device": str(args.device),
            "batch_size": args.batch_size,
            "batches_processed": batches_processed,
            "samples_evaluated": samples_processed,
            "samples_requested": len(dataset),
            "load_errors": load_error_records,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_step": checkpoint.get("step"),
            "confidence_checkpoint": str(args.confidence_checkpoint),
            "confidence_checkpoint_sha256": confidence_hash,
            "confidence_epoch": head_payload.get("epoch"),
            "confidence_step": head_payload.get("step"),
            "confidence_recorded_base_checkpoint": head_payload.get("base_checkpoint"),
            "confidence_recorded_base_sha256": recorded_hash,
            "confidence_pairing_evidence": "stored base path; legacy checkpoint has no historical hash" if recorded_hash is None else "SHA-256 verified",
            "effective_config": config,
            "query_count": query_count,
            "inference_valid_queries": valid_input_queries,
            "satellite_candidates": satellite_candidates,
            "valid_gt_queries": valid_gt_queries,
            "retention_ratios": args.retention_ratios,
            "exact_grid_eligibility_threshold_px": grid_eligibility_px,
            "coarse_geometry_threshold_px": threshold_px,
            "matching_score_mode_for_shared_forward": "forward_softmax",
        },
        "geometric_depth_diagnostic": {
            "metric_role": "geometric_camera_z_matching_diagnostic",
            "canonical_depth_metric": False,
            "scale_correction_applied": False,
            "primary_depth_evaluator": "evaluate_confidence.py",
            "prediction": "camera-z recovered geometrically from the selected north-up satellite match and camera pose",
            "ground_truth": "LiDAR camera-z sampled at front query positions",
            "depth_type": config.get("geometry", {}).get("depth_type", "camera_z"),
            "lidar_query_radius_px": int(config.get("geometry", {}).get("lidar_query_radius_px", 0)),
            "satellite_meters_per_pixel_min": min(mpp_values) if mpp_values else None,
            "satellite_meters_per_pixel_mean": mean(mpp_values) if mpp_values else None,
            "satellite_meters_per_pixel_max": max(mpp_values) if mpp_values else None,
            "metrics": ["AbsRel", "SqRel", "RMSE (m)", "delta1"],
        },
        "depth_evaluation": {
            "metric_protocol": "confidence_weighted_scale_corrected_dense_depth_v1",
            "primary_stage": "refined",
            "stages": ["coarse", "refined"],
            "scale_formula": "exp(sum_i normalized_confidence_i * log(anchor_z_i / mono_anchor_z_i))",
            "weights": "same frozen learned confidence head, sampled at each stage/mode selected location",
            "anchors": "all inference-valid finite positive anchor camera-z <= max_anchor_depth_m and positive mono camera-z; no GT filtering or certainty/top-k gate",
            "mask": "same finite positive mono and configured LiDAR-depth mask for all modes; paired valid-scale frames per stage",
            "aggregation": "mean of per-image metrics",
            "paired_frame_counts": paired_depth_counts,
            "coarse_note": "coarse-stage ablation applies the same head trained on refined anchors at coarse positions",
            "parameters": config.get("confidence", {}),
        },
        "depth_frames": depth_frames,
        "modes": summaries,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "metrics.json"
    output_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "depth_frames"}, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
