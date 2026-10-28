from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .candidate_graph import (
    CandidateCorrespondenceGraph,
    CandidateGeometry,
    select_diverse_candidates,
)


@dataclass
class MatcherOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    coarse_indices: torch.Tensor
    coarse_uv: torch.Tensor
    matchability_logits: torch.Tensor
    contextual_front: torch.Tensor
    contextual_satellite: torch.Tensor
    score_mode: str
    query_valid_mask: torch.Tensor
    satellite_valid_mask: torch.Tensor
    coarse_valid_mask: torch.Tensor
    dual_log_scores: torch.Tensor | None = None
    pre_graph_logits: torch.Tensor | None = None
    candidate_indices: torch.Tensor | None = None
    candidate_valid_mask: torch.Tensor | None = None
    candidate_logits: torch.Tensor | None = None
    candidate_world_xy: torch.Tensor | None = None
    graph_edge_valid_fraction: torch.Tensor | None = None
    graph_scale: torch.Tensor | None = None

    @property
    def score_dual(self) -> torch.Tensor | None:
        """Ordinary bidirectional score, when dual-softmax inference is active."""

        return self.dual_log_scores.exp() if self.dual_log_scores is not None else None

    @property
    def matching_scores(self) -> torch.Tensor:
        """Return the active matching score, separate from matchability/certainty."""

        dual_scores = self.score_dual
        return dual_scores if dual_scores is not None else self.probabilities

    @property
    def selected_matching_scores(self) -> torch.Tensor:
        """Score the selected coarse pair without materializing another matrix."""

        indices = self.coarse_indices.clamp_min(0)
        if self.dual_log_scores is not None:
            selected = self.dual_log_scores.gather(-1, indices[..., None]).squeeze(-1).exp()
        else:
            selected = self.probabilities.gather(-1, indices[..., None]).squeeze(-1)
        return selected.masked_fill(~self.coarse_valid_mask, 0.0)

    @property
    def ranking_scores(self) -> torch.Tensor:
        """Scores used for top-k selection; dual scores stay in log space."""

        if self.dual_log_scores is not None:
            return self.dual_log_scores
        pair_valid = (
            self.query_valid_mask[:, :, None]
            & self.satellite_valid_mask[:, None, :]
            & torch.isfinite(self.logits)
        )
        return self.logits.masked_fill(~pair_valid, -torch.inf)

    def topk(self, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return candidate indices, scores, and validity without padded picks."""

        if k <= 0:
            raise ValueError("k must be positive")
        k = min(k, self.ranking_scores.shape[-1])
        values, indices = self.ranking_scores.topk(k, dim=-1)
        valid = torch.isfinite(values) & self.coarse_valid_mask.unsqueeze(-1)
        return indices.masked_fill(~valid, -1), values, valid


def _masked_log_softmax(logits: torch.Tensor, valid_mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Log-softmax over valid entries, returning -inf for invalid/all-empty rows."""

    masked_logits = logits.masked_fill(~valid_mask, -torch.inf)
    has_value = valid_mask.any(dim=dim, keepdim=True)
    # log_softmax(all -inf) is NaN. Use a harmless row for empty reductions,
    # then restore -inf at every invalid entry.
    safe_logits = torch.where(has_value, masked_logits, torch.zeros_like(masked_logits))
    return torch.log_softmax(safe_logits, dim=dim).masked_fill(~valid_mask, -torch.inf)


def compute_dual_softmax_log_scores(
    logits: torch.Tensor,
    query_valid_mask: torch.Tensor | None = None,
    satellite_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute log(P(front->sat) * P(sat->front)) independently per batch.

    ``logits`` must already include the model's temperature/logit scale. Both
    normalizations run in FP32 and ignore invalid tokens and non-finite pairs.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, N_front, N_satellite]")
    batch_size, num_queries, num_satellites = logits.shape
    if query_valid_mask is None:
        query_valid_mask = torch.ones((batch_size, num_queries), dtype=torch.bool, device=logits.device)
    if satellite_valid_mask is None:
        satellite_valid_mask = torch.ones((batch_size, num_satellites), dtype=torch.bool, device=logits.device)
    if query_valid_mask.shape != (batch_size, num_queries):
        raise ValueError("query_valid_mask must have shape [B, N_front]")
    if satellite_valid_mask.shape != (batch_size, num_satellites):
        raise ValueError("satellite_valid_mask must have shape [B, N_satellite]")

    with torch.autocast(device_type=logits.device.type, enabled=False):
        scores_fp32 = logits.float()
        pair_valid = (
            query_valid_mask.to(device=logits.device, dtype=torch.bool)[:, :, None]
            & satellite_valid_mask.to(device=logits.device, dtype=torch.bool)[:, None, :]
            & torch.isfinite(scores_fp32)
        )
        log_p_f2s = _masked_log_softmax(scores_fp32, pair_valid, dim=-1)
        log_p_s2f = _masked_log_softmax(scores_fp32, pair_valid, dim=-2)
        return log_p_f2s + log_p_s2f


class CrossViewBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_front = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.self_sat = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_front = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_sat = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(6)])
        hidden = int(dim * mlp_ratio)
        self.ff_front = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.ff_sat = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, front: torch.Tensor, sat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        front = front + self.self_front(self.norms[0](front), self.norms[0](front), self.norms[0](front), need_weights=False)[0]
        sat = sat + self.self_sat(self.norms[1](sat), self.norms[1](sat), self.norms[1](sat), need_weights=False)[0]
        front = front + self.cross_front(self.norms[2](front), self.norms[3](sat), self.norms[3](sat), need_weights=False)[0]
        sat = sat + self.cross_sat(self.norms[4](sat), self.norms[5](front), self.norms[5](front), need_weights=False)[0]
        front = front + self.ff_front(front)
        sat = sat + self.ff_sat(sat)
        return front, sat


class CrossViewCoarseMatcher(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        temperature: float = 0.07,
        score_mode: str = "forward_softmax",
        candidate_graph_config: dict | None = None,
    ) -> None:
        super().__init__()
        if score_mode not in {"forward_softmax", "dual_softmax"}:
            raise ValueError("matching.score_mode must be 'forward_softmax' or 'dual_softmax'.")
        self.blocks = nn.ModuleList([CrossViewBlock(dim, num_heads) for _ in range(num_layers)])
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature, dtype=torch.float32)))
        self.score_mode = score_mode
        graph_cfg = candidate_graph_config or {}
        self.candidate_graph_enabled = bool(graph_cfg.get("enabled", False))
        self.candidate_count = int(graph_cfg.get("candidate_count", 4))
        self.candidate_min_separation_px = float(graph_cfg.get("candidate_min_separation_px", 24.0))
        if self.candidate_min_separation_px < 0:
            raise ValueError("matcher.candidate_graph.candidate_min_separation_px must be non-negative")
        self.candidate_graph = CandidateCorrespondenceGraph(
            descriptor_dim=dim,
            candidate_count=self.candidate_count,
            hidden_dim=int(graph_cfg.get("hidden_dim", 96)),
            num_layers=int(graph_cfg.get("num_layers", 2)),
            use_geometry=bool(graph_cfg.get("use_geometry", True)),
            depth_relation=str(graph_cfg.get("depth_relation", "metric_or_global_scale")),
            depth_units=str(graph_cfg.get("depth_units", "arbitrary")),
            depth_scale_mode=str(graph_cfg.get("depth_scale_mode", "robust")),
            include_diagonal_neighbors=bool(graph_cfg.get("include_diagonal_neighbors", True)),
            cross_region_links=bool(graph_cfg.get("cross_region_links", True)),
            rotation_ransac_threshold_rad=float(graph_cfg.get("rotation_ransac_threshold_rad", 0.5)),
            rotation_ransac_hypotheses=int(graph_cfg.get("rotation_ransac_hypotheses", 32)),
            max_logit_residual=float(graph_cfg.get("max_logit_residual", 5.0)),
            dropout=float(graph_cfg.get("dropout", 0.0)),
        )
        self.matchability_head = nn.Sequential(
            nn.LayerNorm(dim + 3),
            nn.Linear(dim + 3, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Load pre-rename checkpoints whose head was called ``confidence_head``."""

        legacy_prefix = f"{prefix}confidence_head."
        current_prefix = f"{prefix}matchability_head."
        for key in list(state_dict):
            if key.startswith(legacy_prefix):
                replacement = f"{current_prefix}{key[len(legacy_prefix):]}"
                state_dict.setdefault(replacement, state_dict[key])
                del state_dict[key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        front_tokens: torch.Tensor,
        sat_tokens: torch.Tensor,
        satellite_grid_uv: torch.Tensor,
        query_valid_mask: torch.Tensor | None = None,
        satellite_valid_mask: torch.Tensor | None = None,
        query_grid_hw: tuple[int, int] | None = None,
        satellite_grid_hw: tuple[int, int] | None = None,
        candidate_geometry: CandidateGeometry | None = None,
    ) -> MatcherOutput:
        front = front_tokens
        sat = sat_tokens
        for block in self.blocks:
            front, sat = block(front, sat)
        front_n = F.normalize(front, dim=-1)
        sat_n = F.normalize(sat, dim=-1)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        logits = torch.matmul(front_n, sat_n.transpose(1, 2)) * scale
        batch_size, num_queries, num_satellites = logits.shape
        if query_valid_mask is None:
            query_valid_mask = torch.ones((batch_size, num_queries), dtype=torch.bool, device=logits.device)
        if satellite_valid_mask is None:
            satellite_valid_mask = torch.ones((batch_size, num_satellites), dtype=torch.bool, device=logits.device)
        if query_valid_mask.shape != (batch_size, num_queries):
            raise ValueError("query_valid_mask must have shape [B, N_front]")
        if satellite_valid_mask.shape != (batch_size, num_satellites):
            raise ValueError("satellite_valid_mask must have shape [B, N_satellite]")
        query_valid_mask = query_valid_mask.to(device=logits.device, dtype=torch.bool) & torch.isfinite(front).all(dim=-1)
        satellite_valid_mask = satellite_valid_mask.to(device=logits.device, dtype=torch.bool) & torch.isfinite(sat).all(dim=-1)
        pre_graph_logits = logits
        pair_valid = (
            query_valid_mask[:, :, None]
            & satellite_valid_mask[:, None, :]
            & torch.isfinite(logits)
        )
        candidate_indices = None
        candidate_valid = None
        graph_edge_valid_fraction = None
        graph_scale = None
        if self.candidate_graph_enabled:
            if satellite_grid_hw is None or query_grid_hw is None:
                raise ValueError("query_grid_hw and satellite_grid_hw are required when candidate graph is enabled")
            candidate_indices, candidate_valid = select_diverse_candidates(
                logits=logits,
                pair_valid=pair_valid,
                satellite_grid_uv=satellite_grid_uv,
                grid_hw=satellite_grid_hw,
                count=self.candidate_count,
                min_separation_px=self.candidate_min_separation_px,
            )
            raw_masked_logits = logits.masked_fill(~pair_valid, -torch.inf)
            raw_forward_prob = torch.softmax(
                torch.where(
                    pair_valid.any(dim=-1, keepdim=True),
                    raw_masked_logits,
                    torch.zeros_like(raw_masked_logits),
                ),
                dim=-1,
            ).masked_fill(~pair_valid, 0.0)
            residual, graph_edge_valid_fraction, graph_scale = self.candidate_graph(
                front=front,
                sat=sat,
                base_logits=logits,
                base_probabilities=raw_forward_prob,
                candidate_indices=candidate_indices,
                candidate_valid=candidate_valid,
                grid_hw=query_grid_hw,
                geometry=candidate_geometry,
            )
            logits = logits.scatter_add(2, candidate_indices.clamp_min(0), residual)
            pair_valid = (
                query_valid_mask[:, :, None]
                & satellite_valid_mask[:, None, :]
                & torch.isfinite(logits)
            )
        masked_logits = logits.masked_fill(~pair_valid, -torch.inf)
        has_forward_candidate = pair_valid.any(dim=-1)
        safe_forward_logits = torch.where(
            has_forward_candidate[..., None], masked_logits, torch.zeros_like(masked_logits)
        )
        probabilities = F.softmax(safe_forward_logits, dim=-1).masked_fill(~pair_valid, 0.0)

        # Training always keeps the original forward-only selection. The
        # configured dual score is used only after model.eval().
        active_score_mode = self.score_mode if not self.training else "forward_softmax"
        dual_log_scores = (
            compute_dual_softmax_log_scores(logits, query_valid_mask, satellite_valid_mask)
            if active_score_mode == "dual_softmax"
            else None
        )
        rank_scores = dual_log_scores if dual_log_scores is not None else masked_logits
        coarse_valid_mask = query_valid_mask & has_forward_candidate
        coarse_indices = rank_scores.argmax(dim=-1).masked_fill(~coarse_valid_mask, -1)
        gather_indices = coarse_indices.clamp_min(0)
        coarse_uv = satellite_grid_uv.to(logits.device, logits.dtype)[gather_indices]
        coarse_uv = coarse_uv.masked_fill(~coarse_valid_mask[..., None], torch.nan)

        safe_matchability_logits = torch.where(
            has_forward_candidate[..., None], masked_logits, torch.zeros_like(masked_logits)
        )
        top2 = safe_matchability_logits.topk(k=min(2, logits.shape[-1]), dim=-1).values
        top1 = top2[..., 0]
        if top2.shape[-1] > 1:
            margin = top2[..., 0] - top2[..., 1]
            margin = torch.where(pair_valid.sum(dim=-1) > 1, margin, torch.zeros_like(margin))
        else:
            margin = torch.zeros_like(top1)
        top1 = torch.where(coarse_valid_mask, top1, torch.zeros_like(top1))
        margin = torch.where(coarse_valid_mask, margin, torch.zeros_like(margin))
        entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        entropy = torch.where(coarse_valid_mask, entropy, torch.zeros_like(entropy))
        matchability_features = torch.cat(
            (front, top1[..., None], margin[..., None], entropy[..., None]), dim=-1
        )
        matchability_logits = self.matchability_head(matchability_features).squeeze(-1)
        return MatcherOutput(
            logits=logits,
            probabilities=probabilities,
            coarse_indices=coarse_indices,
            coarse_uv=coarse_uv,
            matchability_logits=matchability_logits,
            contextual_front=front,
            contextual_satellite=sat,
            score_mode=active_score_mode,
            query_valid_mask=query_valid_mask,
            satellite_valid_mask=satellite_valid_mask,
            coarse_valid_mask=coarse_valid_mask,
            dual_log_scores=dual_log_scores,
            pre_graph_logits=pre_graph_logits if self.candidate_graph_enabled else None,
            candidate_indices=candidate_indices,
            candidate_valid_mask=candidate_valid,
            candidate_logits=(
                logits.gather(-1, candidate_indices.clamp_min(0)) if candidate_indices is not None else None
            ),
            candidate_world_xy=(
                candidate_geometry.satellite_xy[:, None, :, :]
                .expand(-1, num_queries, -1, -1)
                .gather(2, candidate_indices.clamp_min(0)[..., None].expand(-1, -1, -1, 2))
                if candidate_geometry is not None and candidate_indices is not None else None
            ),
            graph_edge_valid_fraction=graph_edge_valid_fraction,
            graph_scale=graph_scale,
        )
