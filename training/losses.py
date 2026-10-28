from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..models.model import Stage1Output
from ..models.candidate_graph import build_sparse_query_pairs


@dataclass
class LossOutput:
    total: torch.Tensor
    coarse: torch.Tensor
    refine: torch.Tensor
    matchability_loss: torch.Tensor
    certainty_loss: torch.Tensor
    graph_candidate_loss: torch.Tensor
    graph_structure_loss: torch.Tensor
    refine_structure_loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def _zero_like(logits: torch.Tensor) -> torch.Tensor:
    return logits.new_zeros(())


def _expected_calibration_error(
    probability: torch.Tensor,
    target: torch.Tensor,
    num_bins: int = 10,
) -> torch.Tensor:
    """Compute ECE for a probability/target pair.

    This is a diagnostic only.  The returned value is detached from the model
    graph and is not used by the training objective.
    """

    if probability.numel() == 0:
        return probability.new_zeros(())
    probability = probability.detach().float().clamp(0.0, 1.0)
    target = target.detach().float()
    ece = probability.new_zeros(())
    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        if bin_index == num_bins - 1:
            in_bin = (probability >= lower) & (probability <= upper)
        else:
            in_bin = (probability >= lower) & (probability < upper)
        if in_bin.any():
            ece = ece + in_bin.float().mean() * (
                probability[in_bin].mean() - target[in_bin].mean()
            ).abs()
    return ece


def compute_losses(output: Stage1Output, config: dict) -> LossOutput:
    loss_cfg = config["loss"]
    stage = str(config.get("training", {}).get("stage", "joint"))
    if stage not in {"coarse", "refine", "joint"}:
        raise ValueError("training.stage must be one of: coarse, refine, joint.")
    logits = output.matcher.logits
    gt_uv = output.targets.gt_sat_uv
    sat_grid = output.satellite_grid_uv.to(gt_uv.device, gt_uv.dtype)
    dist = torch.cdist(gt_uv.reshape(-1, 1, 2), sat_grid.reshape(1, -1, 2)).squeeze(1)
    gt_indices = dist.argmin(dim=-1).reshape(gt_uv.shape[:2])
    coarse_gt_dist = dist.min(dim=-1).values.reshape(gt_uv.shape[:2])
    coarse_loss_valid = output.targets.valid_gt_mask & (
        coarse_gt_dist <= float(loss_cfg.get("coarse_max_gt_grid_dist_px", 32.0))
    )
    coarse_valid = coarse_loss_valid & output.matcher.coarse_valid_mask

    if coarse_loss_valid.any():
        # Preserve the original one-way CE mask and raw scaled logits. Matcher
        # validity only guards inference diagnostics, never changes this loss.
        coarse = F.cross_entropy(logits[coarse_loss_valid], gt_indices[coarse_loss_valid])
    else:
        coarse = _zero_like(logits)

    graph_candidate_loss = _zero_like(logits)
    graph_structure_loss = _zero_like(logits)
    candidate_covered = torch.zeros_like(coarse_loss_valid)
    query_height, query_width = output.front_features.shape[-2:]
    graph_cfg = config.get("matcher", {}).get("candidate_graph", {})
    graph_pairs = build_sparse_query_pairs(
        query_height,
        query_width,
        logits.device,
        include_diagonals=bool(graph_cfg.get("include_diagonal_neighbors", True)),
        include_cross_region=bool(graph_cfg.get("cross_region_links", True)),
    )
    candidate_logits = output.matcher.candidate_logits
    candidate_indices = output.matcher.candidate_indices
    candidate_mask = output.matcher.candidate_valid_mask
    candidate_world_xy = output.matcher.candidate_world_xy
    if candidate_logits is not None and candidate_indices is not None and candidate_mask is not None:
        candidate_covered = (
            (candidate_indices == gt_indices[..., None]) & candidate_mask
        ).any(dim=-1) & coarse_loss_valid
        if candidate_covered.any() and stage in {"coarse", "joint"}:
            safe_candidate_logits = candidate_logits.masked_fill(~candidate_mask, -torch.inf)
            candidate_ranks = (candidate_indices == gt_indices[..., None]).to(torch.long).argmax(dim=-1)
            graph_candidate_loss = F.cross_entropy(
                safe_candidate_logits[candidate_covered], candidate_ranks[candidate_covered]
            )
        if (
            candidate_world_xy is not None
            and graph_pairs.numel() > 0
            and candidate_covered.any()
            and stage in {"coarse", "joint"}
        ):
            safe_candidate_logits = candidate_logits.masked_fill(~candidate_mask, -torch.inf)
            candidate_prob = torch.softmax(
                torch.where(
                    candidate_mask.any(dim=-1, keepdim=True),
                    safe_candidate_logits,
                    torch.zeros_like(safe_candidate_logits),
                ),
                dim=-1,
            ).masked_fill(~candidate_mask, 0.0)
            expected_world_xy = (candidate_prob[..., None] * candidate_world_xy).sum(dim=2)
            first, second = graph_pairs.unbind(dim=-1)
            target_delta = output.targets.gt_world_xy[:, first] - output.targets.gt_world_xy[:, second]
            predicted_delta = expected_world_xy[:, first] - expected_world_xy[:, second]
            pair_valid = candidate_covered[:, first] & candidate_covered[:, second]
            pair_valid &= output.targets.valid_gt_mask[:, first] & output.targets.valid_gt_mask[:, second]
            pair_valid &= output.satellite_geometry_valid[:, None]
            min_baseline = float(loss_cfg.get("graph_structure_min_baseline_m", 1.0))
            pair_valid &= torch.linalg.vector_norm(target_delta, dim=-1) >= min_baseline
            pair_valid &= torch.isfinite(predicted_delta).all(dim=-1) & torch.isfinite(target_delta).all(dim=-1)
            if pair_valid.any():
                beta = float(loss_cfg.get("graph_structure_huber_beta_m", 1.0))
                graph_structure_loss = F.smooth_l1_loss(
                    predicted_delta[pair_valid], target_delta[pair_valid], beta=beta
                )

    if output.targets.valid_gt_mask.any():
        refine_error = torch.linalg.norm(output.refined_uv - gt_uv, dim=-1)
        refine = F.smooth_l1_loss(output.refined_uv[output.targets.valid_gt_mask], gt_uv[output.targets.valid_gt_mask])
    else:
        refine_error = torch.zeros_like(output.targets.valid_gt_mask, dtype=logits.dtype)
        refine = _zero_like(logits)

    refine_structure_loss = _zero_like(logits)
    if (
        stage in {"refine", "joint"}
        and output.matcher.candidate_indices is not None
        and graph_pairs.numel() > 0
    ):
        first, second = graph_pairs.unbind(dim=-1)
        target_delta = output.targets.gt_world_xy[:, first] - output.targets.gt_world_xy[:, second]
        predicted_delta = output.refined_world_xy[:, first] - output.refined_world_xy[:, second]
        pair_valid = output.targets.valid_gt_mask[:, first] & output.targets.valid_gt_mask[:, second]
        pair_valid &= output.matcher.coarse_valid_mask[:, first] & output.matcher.coarse_valid_mask[:, second]
        pair_valid &= output.satellite_geometry_valid[:, None]
        pair_valid &= torch.linalg.vector_norm(target_delta, dim=-1) >= float(
            loss_cfg.get("refine_structure_min_baseline_m", loss_cfg.get("graph_structure_min_baseline_m", 1.0))
        )
        pair_valid &= torch.isfinite(predicted_delta).all(dim=-1) & torch.isfinite(target_delta).all(dim=-1)
        if pair_valid.any():
            refine_structure_loss = F.smooth_l1_loss(
                predicted_delta[pair_valid], target_delta[pair_valid],
                beta=float(loss_cfg.get("refine_structure_huber_beta_m", loss_cfg.get("graph_structure_huber_beta_m", 1.0))),
            )

    matchability_supervised = output.targets.matchability_supervised_mask
    matchability_positive = matchability_supervised & (output.targets.matchability_targets > 0.5)
    matchability_negative = matchability_supervised & (output.targets.matchability_targets <= 0.5)
    matchability_unknown = ~matchability_supervised
    if matchability_supervised.any():
        matchability_loss = (
            F.binary_cross_entropy_with_logits(
                output.matcher.matchability_logits[matchability_supervised],
                output.targets.matchability_targets[matchability_supervised],
            )
            if stage == "coarse"
            else _zero_like(logits)
        )
        certainty_loss = (
            F.binary_cross_entropy_with_logits(
                output.certainty_logits[matchability_supervised],
                output.targets.matchability_targets[matchability_supervised],
            )
            if stage in {"refine", "joint"}
            else _zero_like(logits)
        )
    else:
        matchability_loss = _zero_like(logits)
        certainty_loss = _zero_like(logits)
    head_loss = matchability_loss if stage == "coarse" else certainty_loss
    certainty_weight = float(
        loss_cfg.get("lambda_certainty", loss_cfg.get("lambda_matchability", 0.1))
    )

    if stage == "coarse":
        total = (
            float(loss_cfg.get("lambda_coarse", 1.0)) * coarse
            + float(loss_cfg.get("lambda_matchability", 0.1)) * matchability_loss
            + float(loss_cfg.get("lambda_graph_candidate", 0.0)) * graph_candidate_loss
            + float(loss_cfg.get("lambda_graph_structure", 0.0)) * graph_structure_loss
        )
    elif stage == "refine":
        total = (
            float(loss_cfg.get("lambda_refine", 1.0)) * refine
            + certainty_weight * certainty_loss
            + float(loss_cfg.get("lambda_refine_structure", 0.0)) * refine_structure_loss
        )
    else:
        total = (
            float(loss_cfg.get("lambda_coarse", 1.0)) * coarse
            + float(loss_cfg.get("lambda_refine", 1.0)) * refine
            + certainty_weight * certainty_loss
            + float(loss_cfg.get("lambda_graph_candidate", 0.0)) * graph_candidate_loss
            + float(loss_cfg.get("lambda_graph_structure", 0.0)) * graph_structure_loss
            + float(loss_cfg.get("lambda_refine_structure", 0.0)) * refine_structure_loss
        )

    pred_top1 = output.matcher.coarse_indices
    metrics: dict[str, torch.Tensor] = {
        "num_queries": torch.tensor(float(gt_uv.shape[1]), device=logits.device),
        "num_valid_gt": output.targets.valid_gt_mask.sum().float(),
        "num_matchability_positive": matchability_positive.sum().float(),
        "num_matchability_negative": matchability_negative.sum().float(),
        "num_matchability_unknown": matchability_unknown.sum().float(),
        "loss_total": total.detach(),
        "loss_coarse": coarse.detach(),
        "loss_refine": refine.detach(),
        "loss_graph_candidate": graph_candidate_loss.detach(),
        "loss_graph_structure": graph_structure_loss.detach(),
        "loss_refine_structure": refine_structure_loss.detach(),
        "num_candidate_covered": candidate_covered.sum().float(),
        "candidate_coverage": (
            candidate_covered.sum().float() / coarse_loss_valid.sum().clamp_min(1).float()
        ),
    }
    if output.matcher.graph_edge_valid_fraction is not None:
        metrics["graph_geometry_edge_valid_fraction"] = output.matcher.graph_edge_valid_fraction.mean().detach()
    if output.matcher.graph_scale is not None:
        metrics["graph_shared_scale"] = output.matcher.graph_scale.mean().detach()
    metrics["loss_matchability" if stage == "coarse" else "loss_certainty"] = head_loss.detach()
    if coarse_valid.any():
        coarse_pred = pred_top1[coarse_valid]
        coarse_gt = gt_indices[coarse_valid]
        coarse_probabilities = output.matcher.probabilities[coarse_valid]
        forward_pred = logits.argmax(dim=-1)[coarse_valid]
        coarse_max_probability = coarse_probabilities.max(dim=-1).values
        coarse_gt_probability = coarse_probabilities.gather(1, coarse_gt[:, None]).squeeze(1)
        coarse_correct = coarse_pred == coarse_gt
        forward_correct = forward_pred == coarse_gt
        coarse_entropy = -(
            coarse_probabilities.clamp_min(1e-12)
            * coarse_probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1)
        selected_scores = (
            output.matcher.dual_log_scores[coarse_valid].exp()
            if output.matcher.dual_log_scores is not None
            else coarse_probabilities
        )
        selected_max_score = selected_scores.max(dim=-1).values
        selected_gt_score = selected_scores.gather(1, coarse_gt[:, None]).squeeze(1)
        entropy_denominator = torch.log(
            coarse_probabilities.new_tensor(float(max(coarse_probabilities.shape[-1], 2)))
        )

        metrics["num_coarse_valid"] = torch.tensor(float(coarse_valid.sum()), device=logits.device)
        metrics["coarse_top1_accuracy"] = coarse_correct.float().mean()
        metrics["coarse_gt_distance"] = coarse_gt_dist[coarse_valid].mean()
        # These calibration diagnostics always describe the original forward
        # softmax and its own argmax, even when inference selects by dual score.
        metrics["coarse_match_gt_prob"] = coarse_gt_probability.mean().detach()
        metrics["coarse_match_max_prob"] = coarse_max_probability.mean().detach()
        metrics["coarse_match_entropy_norm"] = (coarse_entropy / entropy_denominator).mean().detach()
        metrics["coarse_match_ece"] = _expected_calibration_error(
            coarse_max_probability,
            forward_correct,
        )
        metrics["coarse_match_brier"] = (
            (coarse_max_probability - forward_correct.float()).square().mean().detach()
        )
        metrics["coarse_selected_max_score"] = selected_max_score.mean().detach()
        metrics["coarse_selected_gt_score"] = selected_gt_score.mean().detach()
        metrics["coarse_match_score_is_dual"] = torch.tensor(
            float(output.matcher.score_mode == "dual_softmax"), device=logits.device
        )
        if forward_correct.any():
            metrics["coarse_match_correct_max_prob"] = coarse_max_probability[forward_correct].mean().detach()
        else:
            metrics["coarse_match_correct_max_prob"] = _zero_like(logits)
        if (~forward_correct).any():
            metrics["coarse_match_wrong_max_prob"] = coarse_max_probability[~forward_correct].mean().detach()
        else:
            metrics["coarse_match_wrong_max_prob"] = _zero_like(logits)
    else:
        metrics["num_coarse_valid"] = _zero_like(logits)
        metrics["coarse_top1_accuracy"] = _zero_like(logits)
        metrics["coarse_gt_distance"] = _zero_like(logits)
        metrics["coarse_match_gt_prob"] = _zero_like(logits)
        metrics["coarse_match_max_prob"] = _zero_like(logits)
        metrics["coarse_match_entropy_norm"] = _zero_like(logits)
        metrics["coarse_match_ece"] = _zero_like(logits)
        metrics["coarse_match_brier"] = _zero_like(logits)
        metrics["coarse_match_correct_max_prob"] = _zero_like(logits)
        metrics["coarse_match_wrong_max_prob"] = _zero_like(logits)

    def add_probability_metrics(prefix: str, probability: torch.Tensor) -> None:
        """Add calibration metrics for one binary probability head.

        ``matchability_targets`` are the labels used by both the matchability
        head and the certainty head. Keeping the prefix explicit prevents the
        two heads from being mixed in TensorBoard.
        """

        if matchability_supervised.any():
            probability = probability[matchability_supervised]
            target = output.targets.matchability_targets[matchability_supervised]
            correct = (probability >= 0.5) == (target >= 0.5)
            metrics[f"{prefix}_mean"] = probability.mean().detach()
            metrics[f"{prefix}_accuracy"] = correct.float().mean().detach()
            metrics[f"{prefix}_ece"] = _expected_calibration_error(probability, target)
            metrics[f"{prefix}_brier"] = (probability - target).square().mean().detach()
            positive = matchability_positive[matchability_supervised]
            negative = matchability_negative[matchability_supervised]
            metrics[f"{prefix}_positive_mean"] = (
                probability[positive].mean().detach() if positive.any() else _zero_like(logits)
            )
            metrics[f"{prefix}_negative_mean"] = (
                probability[negative].mean().detach() if negative.any() else _zero_like(logits)
            )
        else:
            for suffix in ("mean", "accuracy", "ece", "brier", "positive_mean", "negative_mean"):
                metrics[f"{prefix}_{suffix}"] = _zero_like(logits)

    # The matchability head is read from the matcher before refine deltas are
    # added. The final output is logged separately as certainty.
    add_probability_metrics(
        "matchability",
        output.matcher.matchability_logits.sigmoid(),
    )

    # The final output is the certainty in refine/joint stages. It is trained
    # against the same matchability labels through the final logits.
    if stage in {"refine", "joint"}:
        add_probability_metrics(
            "certainty",
            output.certainty_logits.sigmoid(),
        )
    if output.targets.valid_gt_mask.any():
        err = refine_error[output.targets.valid_gt_mask]
        mpp = output.satellite_meters_per_pixel[:, None].expand_as(refine_error)[output.targets.valid_gt_mask]
        metrics["refine_pixel_error"] = err.mean()
        metrics["median_refine_pixel_error"] = err.median()
        metrics["mean_metric_error"] = (err * mpp).mean()
        metrics["median_metric_error"] = (err * mpp).median()
    else:
        metrics["refine_pixel_error"] = _zero_like(logits)
        metrics["median_refine_pixel_error"] = _zero_like(logits)
        metrics["mean_metric_error"] = _zero_like(logits)
        metrics["median_metric_error"] = _zero_like(logits)
    return LossOutput(
        total=total,
        coarse=coarse,
        refine=refine,
        matchability_loss=matchability_loss,
        certainty_loss=certainty_loss,
        graph_candidate_loss=graph_candidate_loss,
        graph_structure_loss=graph_structure_loss,
        refine_structure_loss=refine_structure_loss,
        metrics=metrics,
    )
