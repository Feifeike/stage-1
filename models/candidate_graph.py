from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CandidateGeometry:
    """Geometry for coarse front queries and satellite candidate locations.

    ``front_xy`` is the horizontal component of the 3D point in the calibrated
    local vehicle frame (meters, or a common arbitrary monocular-depth unit).
    ``satellite_xy`` is in map-world XY and meters. A single proper rotation
    and positive scale are estimated from predicted correspondences; absolute
    vehicle pose and translation are never read here.
    """

    front_xy: torch.Tensor  # [B, N, 2]
    front_valid: torch.Tensor  # [B, N]
    front_reliability: torch.Tensor  # [B, N], [0, 1]
    satellite_xy: torch.Tensor  # [B, M, 2]


def build_sparse_query_pairs(
    height: int,
    width: int,
    device: torch.device,
    *,
    include_diagonals: bool = True,
    include_cross_region: bool = True,
) -> torch.Tensor:
    """Build undirected local plus cross-region query links, without self edges."""

    if height <= 0 or width <= 0:
        raise ValueError("query grid dimensions must be positive")
    ids = torch.arange(height * width, device=device).reshape(height, width)
    pairs: list[torch.Tensor] = []
    # Cardinal edges give local support; optional diagonals and half-grid
    # shifts add broader local/cross-region information at bounded degree.
    offsets = [(0, 1), (1, 0)]
    if include_diagonals:
        offsets.extend(((1, 1), (1, -1)))
    if include_cross_region:
        if height > 2:
            offsets.append((max(1, height // 2), 0))
        if width > 2:
            offsets.append((0, max(1, width // 2)))
    for dy, dx in offsets:
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        if y1 <= y0 or x1 <= x0:
            continue
        first = ids[y0:y1, x0:x1].reshape(-1)
        second = ids[y0 + dy:y1 + dy, x0 + dx:x1 + dx].reshape(-1)
        keep = first != second
        pairs.append(torch.stack((first[keep], second[keep]), dim=-1))
    if not pairs:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    return torch.unique(torch.cat(pairs, dim=0), dim=0)


def make_candidate_edges(
    query_pairs: torch.Tensor,
    candidate_count: int,
    candidate_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand sparse query links to directed candidate-node links.

    Candidate alternatives belonging to the same front query are never
    connected, so they compete only through the within-query normalization.
    """

    num_queries = candidate_valid.shape[0]
    if query_pairs.numel() == 0 or candidate_count <= 0:
        empty = torch.empty((0,), device=candidate_valid.device, dtype=torch.long)
        return empty, empty
    rank = torch.arange(candidate_count, device=candidate_valid.device)
    first = query_pairs[:, 0]
    second = query_pairs[:, 1]
    src = (first[:, None, None] * candidate_count + rank[None, :, None]).expand(-1, -1, candidate_count)
    dst = (second[:, None, None] * candidate_count + rank[None, None, :]).expand(-1, candidate_count, -1)
    src, dst = src.reshape(-1), dst.reshape(-1)
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    valid_flat = candidate_valid.reshape(num_queries * candidate_count)
    keep = valid_flat[src] & valid_flat[dst]
    return src[keep], dst[keep]


def geometry_edge_features(
    front_xy: torch.Tensor,
    front_valid: torch.Tensor,
    front_reliability: torch.Tensor,
    candidate_xy: torch.Tensor,
    candidate_valid: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    candidate_count: int,
    shared_scale: torch.Tensor | float = 1.0,
    shared_rotation: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode common-scale, common-world-frame pair geometry.

    The horizontal coordinate axes already share the world frame after pose
    rotation and satellite-axis conversion, so the justified plane rotation is
    identity (a proper, non-reflecting rotation). ``shared_scale`` is one
    positive value per image, never an edge-specific fit.

    Returns edge features ``[cos(angle), sin(angle), log distance ratio,
    normalized residual, reliability, valid]``, a compatibility score, and a
    per-edge validity mask. Distances are only interpreted when both endpoint
    depths are valid.
    """

    if src.numel() == 0:
        empty_features = front_xy.new_empty((0, 6))
        empty = front_xy.new_empty((0,))
        return empty_features, empty, torch.empty((0,), device=front_xy.device, dtype=torch.bool)
    query_i = torch.div(src, candidate_count, rounding_mode="floor")
    query_j = torch.div(dst, candidate_count, rounding_mode="floor")
    rank_i = src.remainder(candidate_count)
    rank_j = dst.remainder(candidate_count)
    b_delta = front_xy[query_i] - front_xy[query_j]
    q_delta = candidate_xy[query_i, rank_i] - candidate_xy[query_j, rank_j]
    geometry_valid = front_valid[query_i] & front_valid[query_j]
    geometry_valid &= candidate_valid[query_i, rank_i] & candidate_valid[query_j, rank_j]
    geometry_valid &= torch.isfinite(b_delta).all(dim=-1) & torch.isfinite(q_delta).all(dim=-1)

    b_norm = torch.linalg.vector_norm(b_delta, dim=-1)
    q_norm = torch.linalg.vector_norm(q_delta, dim=-1)
    geometry_valid &= (b_norm > eps) & (q_norm > eps)
    if not torch.is_tensor(shared_scale):
        shared_scale = front_xy.new_tensor(float(shared_scale))
    scale = shared_scale.to(device=front_xy.device, dtype=front_xy.dtype).clamp_min(eps)
    if shared_rotation is None:
        shared_rotation = torch.eye(2, device=front_xy.device, dtype=front_xy.dtype)
    rotated_b = torch.matmul(b_delta, shared_rotation.to(front_xy).transpose(-1, -2))
    scaled_b = rotated_b * scale
    dot = (scaled_b * q_delta).sum(dim=-1)
    cross = scaled_b[..., 0] * q_delta[..., 1] - scaled_b[..., 1] * q_delta[..., 0]
    angle = torch.atan2(cross, dot)
    residual = torch.linalg.vector_norm(q_delta - scaled_b, dim=-1) / (q_norm + scale * b_norm + eps)
    log_ratio = torch.log((q_norm + eps) / (scale * b_norm + eps))
    reliability = front_reliability[query_i] * front_reliability[query_j]
    valid_float = geometry_valid.to(front_xy.dtype)
    values = torch.stack(
        (
            torch.cos(angle),
            torch.sin(angle),
            log_ratio.clamp(-8.0, 8.0),
            residual.clamp(0.0, 2.0),
            reliability,
            valid_float,
        ),
        dim=-1,
    )
    values = values * valid_float[:, None]
    compatibility = torch.exp(-residual / 0.15) * reliability * valid_float
    return values, compatibility, geometry_valid


def estimate_shared_similarity(
    front_xy: torch.Tensor,
    front_valid: torch.Tensor,
    top_candidate_xy: torch.Tensor,
    top_candidate_valid: torch.Tensor,
    query_pairs: torch.Tensor,
    mode: str,
    ransac_threshold_rad: float = 0.5,
    max_hypotheses: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Fit one proper 2D rotation and positive scale from predicted pairs.

    A small deterministic RANSAC over pairwise direction angles rejects bad
    top-1 correspondences. No target correspondences or vehicle world pose are
    used. Metric mode fixes scale to one but still estimates the shared frame
    rotation from predictions.
    """

    identity = torch.eye(2, device=front_xy.device, dtype=front_xy.dtype)
    unit_scale = front_xy.new_tensor(1.0)
    if mode not in {"metric", "robust"}:
        raise ValueError("candidate_graph.depth_scale_mode must be 'metric' or 'robust'.")
    if query_pairs.numel() == 0:
        return unit_scale, identity, False
    first, second = query_pairs.unbind(dim=-1)
    b_delta = front_xy[first] - front_xy[second]
    q_delta = top_candidate_xy[first] - top_candidate_xy[second]
    b_norm = torch.linalg.vector_norm(b_delta, dim=-1)
    q_norm = torch.linalg.vector_norm(q_delta, dim=-1)
    valid = front_valid[first] & front_valid[second]
    valid &= top_candidate_valid[first] & top_candidate_valid[second]
    valid &= torch.isfinite(b_norm) & torch.isfinite(q_norm) & (b_norm > 1e-5) & (q_norm > 1e-5)
    b_delta = b_delta[valid]
    q_delta = q_delta[valid]
    b_norm_sq = b_norm[valid].square()
    if b_delta.shape[0] < 2:
        return unit_scale, identity, False

    cross = b_delta[:, 0] * q_delta[:, 1] - b_delta[:, 1] * q_delta[:, 0]
    dot = (b_delta * q_delta).sum(dim=-1)
    angles = torch.atan2(cross, dot)
    # Hypothesize from a bounded, evenly spaced edge sample. This keeps the
    # robust frame fit linear in the number of sparse query links.
    num_hypotheses = min(max(1, int(max_hypotheses)), angles.numel())
    hypothesis_indices = torch.linspace(
        0, angles.numel() - 1, steps=num_hypotheses, device=angles.device
    ).round().long()
    hypotheses = angles[hypothesis_indices]
    angle_delta = angles[None, :] - hypotheses[:, None]
    angle_delta = torch.atan2(torch.sin(angle_delta), torch.cos(angle_delta)).abs()
    inlier_counts = (angle_delta < float(ransac_threshold_rad)).sum(dim=-1)
    best = inlier_counts.argmax()
    inliers = angle_delta[best] < float(ransac_threshold_rad)
    if int(inliers.sum().item()) < 2:
        return unit_scale, identity, False
    inlier_angles = angles[inliers]
    theta = torch.atan2(torch.sin(inlier_angles).mean(), torch.cos(inlier_angles).mean())
    cosine, sine = torch.cos(theta), torch.sin(theta)
    rotation = torch.stack((torch.stack((cosine, -sine)), torch.stack((sine, cosine))))
    rotated_b = b_delta[inliers] @ rotation.transpose(0, 1)
    projected_scale = (q_delta[inliers] * rotated_b).sum(dim=-1) / b_norm_sq[inliers].clamp_min(1e-8)
    projected_scale = projected_scale[torch.isfinite(projected_scale) & (projected_scale > 1e-4)]
    if projected_scale.numel() == 0:
        return unit_scale, identity, False
    scale = unit_scale if mode == "metric" else projected_scale.median().clamp(1e-4, 1e4)
    # Rotation and scale are nuisance estimates from current predictions; do
    # not backpropagate through the discrete robust estimator.
    return scale.detach(), rotation.detach(), True


class CandidateGraphMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1)
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        if src.numel() == 0:
            return nodes
        edge = self.edge_encoder(edge_features)
        source = nodes[src]
        target = nodes[dst]
        attention = torch.sigmoid(self.attention(torch.cat((source, target, edge), dim=-1))).to(nodes.dtype)
        messages = self.message(torch.cat((source, edge), dim=-1)) * attention
        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(0, dst, messages)
        degree = nodes.new_zeros((nodes.shape[0], 1))
        degree.index_add_(0, dst, attention)
        aggregate = aggregate / degree.clamp_min(1e-6)
        return self.norm(nodes + self.update(torch.cat((nodes, aggregate), dim=-1)))


class CandidateCorrespondenceGraph(nn.Module):
    """Sparse MPNN which refines the scores of diverse coarse candidates."""

    def __init__(
        self,
        descriptor_dim: int,
        *,
        candidate_count: int = 4,
        hidden_dim: int = 96,
        num_layers: int = 2,
        use_geometry: bool = True,
        depth_relation: str = "metric_or_global_scale",
        depth_units: str = "arbitrary",
        depth_scale_mode: str = "robust",
        include_diagonal_neighbors: bool = True,
        cross_region_links: bool = True,
        rotation_ransac_threshold_rad: float = 0.5,
        rotation_ransac_hypotheses: int = 32,
        max_logit_residual: float = 5.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if candidate_count <= 0 or hidden_dim < 2 or num_layers <= 0:
            raise ValueError("candidate_count and num_layers must be positive; hidden_dim must be at least 2")
        if max_logit_residual < 0:
            raise ValueError("max_logit_residual must be non-negative")
        if depth_scale_mode not in {"metric", "robust"}:
            raise ValueError("depth_scale_mode must be 'metric' or 'robust'")
        if depth_relation not in {"metric_or_global_scale", "ranking_only"}:
            raise ValueError("depth_relation must be 'metric_or_global_scale' or 'ranking_only'")
        if depth_units not in {"metric", "arbitrary"}:
            raise ValueError("depth_units must be 'metric' or 'arbitrary'")
        if depth_scale_mode == "metric" and depth_units != "metric":
            raise ValueError("metric depth_scale_mode requires depth_units='metric'.")
        if rotation_ransac_threshold_rad <= 0 or rotation_ransac_hypotheses <= 0:
            raise ValueError("rotation RANSAC threshold and hypothesis count must be positive")
        self.candidate_count = int(candidate_count)
        # Ordinal-only depth does not define meaningful 3D distances or
        # horizontal point layouts. Keep the visual graph active while
        # explicitly suppressing geometric edge features in that mode.
        self.use_geometry = bool(use_geometry) and depth_relation != "ranking_only"
        self.depth_relation = str(depth_relation)
        self.depth_units = str(depth_units)
        self.depth_scale_mode = str(depth_scale_mode)
        self.include_diagonal_neighbors = bool(include_diagonal_neighbors)
        self.cross_region_links = bool(cross_region_links)
        self.rotation_ransac_threshold_rad = float(rotation_ransac_threshold_rad)
        self.rotation_ransac_hypotheses = int(rotation_ransac_hypotheses)
        self.max_logit_residual = float(max_logit_residual)
        edge_dim = 6
        # Four scalar inputs: centered candidate score, log candidate
        # probability, query confidence, and normalized entropy.
        self.node_encoder = nn.Sequential(
            nn.LayerNorm(descriptor_dim * 2 + 4),
            nn.Linear(descriptor_dim * 2 + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [CandidateGraphMessageLayer(hidden_dim, edge_dim, dropout) for _ in range(num_layers)]
        )
        self.score_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        nn.init.normal_(self.score_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.score_head[-1].bias)

    @staticmethod
    def _empty_geometry(front: torch.Tensor, candidate_xy: torch.Tensor, candidate_valid: torch.Tensor):
        n = front.shape[0]
        return (
            front.new_zeros((n, 2)),
            torch.zeros((n,), device=front.device, dtype=torch.bool),
            front.new_zeros((n,)),
            candidate_xy,
            candidate_valid,
        )

    def _one_sample(
        self,
        front: torch.Tensor,
        sat: torch.Tensor,
        raw_logits: torch.Tensor,
        raw_probabilities: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_valid: torch.Tensor,
        query_confidence: torch.Tensor,
        query_entropy: torch.Tensor,
        candidate_xy: torch.Tensor | None,
        geometry: CandidateGeometry | None,
        query_pairs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, k = candidate_indices.shape
        safe_indices = candidate_indices.clamp_min(0)
        candidate_sat = sat[safe_indices]
        candidate_prob = raw_probabilities.gather(1, safe_indices)
        row_mean = (raw_logits * candidate_valid).sum(dim=-1, keepdim=True) / candidate_valid.sum(dim=-1, keepdim=True).clamp_min(1)
        centered_score = raw_logits - row_mean
        score_scale = centered_score.detach().abs().mean(dim=-1, keepdim=True).clamp_min(1.0)
        node_scalars = torch.stack(
            (
                centered_score / score_scale,
                candidate_prob.clamp_min(1e-8).log(),
                query_confidence[:, None].expand(-1, k),
                query_entropy[:, None].expand(-1, k),
            ),
            dim=-1,
        )
        nodes = self.node_encoder(
            torch.cat((front[:, None, :].expand(-1, k, -1), candidate_sat, node_scalars), dim=-1)
        ).reshape(n * k, -1)
        nodes = nodes * candidate_valid.reshape(-1, 1).to(nodes.dtype)
        src, dst = make_candidate_edges(query_pairs, k, candidate_valid)

        edge_features = front.new_zeros((src.numel(), 6))
        geometry_valid_fraction = front.new_zeros(())
        shared_scale = front.new_tensor(1.0)
        if self.use_geometry and geometry is not None and candidate_xy is not None:
            front_xy = geometry.front_xy[0]
            front_valid = geometry.front_valid[0]
            front_reliability = geometry.front_reliability[0]
            # ``candidate_xy`` was gathered from the satellite map using the
            # candidate indices by ``forward`` and is already [N, K, 2].
            selected_xy = candidate_xy
            top_valid = candidate_valid[:, 0]
            top_xy = selected_xy[:, 0]
            shared_scale, shared_rotation, transform_valid = estimate_shared_similarity(
                front_xy,
                front_valid,
                top_xy,
                top_valid,
                query_pairs,
                self.depth_scale_mode,
                self.rotation_ransac_threshold_rad,
                self.rotation_ransac_hypotheses,
            )
            if not transform_valid:
                front_valid = torch.zeros_like(front_valid)
            edge_features, _, edge_geometry_valid = geometry_edge_features(
                front_xy=front_xy,
                front_valid=front_valid,
                front_reliability=front_reliability,
                candidate_xy=selected_xy,
                candidate_valid=candidate_valid,
                src=src,
                dst=dst,
                candidate_count=k,
                shared_scale=shared_scale,
                shared_rotation=shared_rotation,
            )
            geometry_valid_fraction = edge_geometry_valid.float().mean() if edge_geometry_valid.numel() else front.new_zeros(())

        for layer in self.layers:
            nodes = layer(nodes, src, dst, edge_features)
        residual = self.max_logit_residual * torch.tanh(self.score_head(nodes).reshape(n, k))
        residual = residual.masked_fill(~candidate_valid, 0.0)
        candidate_count = candidate_valid.sum(dim=-1, keepdim=True)
        residual_mean = (residual * candidate_valid).sum(dim=-1, keepdim=True) / candidate_count.clamp_min(1)
        residual = (residual - residual_mean).masked_fill(~candidate_valid, 0.0)
        return residual, geometry_valid_fraction, shared_scale

    def forward(
        self,
        *,
        front: torch.Tensor,
        sat: torch.Tensor,
        base_logits: torch.Tensor,
        base_probabilities: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_valid: torch.Tensor,
        grid_hw: tuple[int, int],
        geometry: CandidateGeometry | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n, k = candidate_indices.shape
        if k != self.candidate_count:
            raise ValueError(f"candidate dimension {k} does not match configured {self.candidate_count}")
        query_confidence = base_probabilities.max(dim=-1).values
        entropy = -(base_probabilities.clamp_min(1e-8) * base_probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / torch.log(base_probabilities.new_tensor(float(max(base_probabilities.shape[-1], 2))))
        query_pairs = build_sparse_query_pairs(
            grid_hw[0],
            grid_hw[1],
            front.device,
            include_diagonals=self.include_diagonal_neighbors,
            include_cross_region=self.cross_region_links,
        )
        residuals, valid_fractions, scales = [], [], []
        for batch_index in range(batch_size):
            geom = None
            if geometry is not None:
                geom = CandidateGeometry(
                    front_xy=geometry.front_xy[batch_index:batch_index + 1],
                    front_valid=geometry.front_valid[batch_index:batch_index + 1],
                    front_reliability=geometry.front_reliability[batch_index:batch_index + 1],
                    satellite_xy=geometry.satellite_xy[batch_index:batch_index + 1],
                )
                candidate_xy = geom.satellite_xy[0][candidate_indices[batch_index].clamp_min(0)]
            else:
                candidate_xy = None
            residual, valid_fraction, scale = self._one_sample(
                front=front[batch_index],
                sat=sat[batch_index],
                raw_logits=base_logits[batch_index].gather(1, candidate_indices[batch_index].clamp_min(0)),
                raw_probabilities=base_probabilities[batch_index],
                candidate_indices=candidate_indices[batch_index],
                candidate_valid=candidate_valid[batch_index],
                query_confidence=query_confidence[batch_index],
                query_entropy=entropy[batch_index],
                candidate_xy=candidate_xy,
                geometry=geom,
                query_pairs=query_pairs,
            )
            residuals.append(residual)
            valid_fractions.append(valid_fraction)
            scales.append(scale)
        return torch.stack(residuals), torch.stack(valid_fractions), torch.stack(scales)


def select_diverse_candidates(
    logits: torch.Tensor,
    pair_valid: torch.Tensor,
    satellite_grid_uv: torch.Tensor,
    grid_hw: tuple[int, int],
    count: int,
    min_separation_px: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedily retain strong, spatially separated candidates per front point."""

    batch_size, n, m = logits.shape
    h, w = grid_hw
    if h * w != m:
        raise ValueError("satellite_grid_hw does not match the flattened satellite token count")
    requested_count = int(count)
    count = min(requested_count, m)
    base = logits.masked_fill(~pair_valid, -torch.inf)
    working = base.clone()
    selected_mask = torch.zeros_like(pair_valid)
    selected_indices, selected_valid = [], []
    coords = satellite_grid_uv.to(device=logits.device, dtype=logits.dtype)
    for _ in range(count):
        has_available = torch.isfinite(working).any(dim=-1)
        fallback = base.masked_fill(selected_mask, -torch.inf)
        scores = torch.where(has_available[..., None], working, fallback)
        values, indices = scores.max(dim=-1)
        valid = torch.isfinite(values)
        selected_indices.append(indices)
        selected_valid.append(valid)
        selected_mask.scatter_(-1, indices[..., None], True)
        if min_separation_px > 0:
            selected_uv = coords[indices]
            distance_sq = (coords[None, None, :, :] - selected_uv[:, :, None, :]).square().sum(dim=-1)
            suppress = distance_sq <= float(min_separation_px) ** 2
            working = working.masked_fill(suppress & pair_valid, -torch.inf)
        else:
            working.scatter_(-1, indices[..., None], -torch.inf)
    indices = torch.stack(selected_indices, dim=-1)
    valid = torch.stack(selected_valid, dim=-1)
    if requested_count > count:
        pad = requested_count - count
        indices = F.pad(indices, (0, pad), value=0)
        valid = F.pad(valid, (0, pad), value=False)
    return indices, valid


def build_candidate_geometry(
    *,
    query_uv: torch.Tensor,
    satellite_grid_uv: torch.Tensor,
    depth: torch.Tensor,
    depth_valid: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_vehicle: torch.Tensor,
    vehicle_horizontal_axes: tuple[int, int],
    satellite_meters_per_pixel: torch.Tensor,
    satellite_center_world_xy: torch.Tensor,
    satellite_forward_world_xy: torch.Tensor,
    satellite_right_world_xy: torch.Tensor,
    satellite_is_ego_oriented: torch.Tensor,
    front_image_hw: tuple[int, int],
    satellite_image_hw: tuple[int, int],
    depth_type: str,
    depth_valid_weight_threshold: float = 0.5,
    min_depth_confidence: float = 0.0,
    depth_confidence: torch.Tensor | None = None,
) -> CandidateGeometry:
    """Sample depth and convert points to calibrated local horizontal axes.

    ``camera_to_vehicle`` is a fixed sensor extrinsic, not a world pose. Its
    translation is omitted because it cancels in pairwise differences. The
    selected local axes are mapped to satellite world XY by one robust shared
    rotation estimated from predicted correspondences.
    """

    supported = {"camera_z", "ray_distance", "inverse_camera_z", "inverse_ray_distance"}
    if depth_type not in supported:
        raise ValueError(f"Unsupported candidate_graph.depth_type={depth_type!r}; expected one of {sorted(supported)}")
    if (
        len(vehicle_horizontal_axes) != 2
        or len(set(vehicle_horizontal_axes)) != 2
        or any(axis not in (0, 1, 2) for axis in vehicle_horizontal_axes)
    ):
        raise ValueError("vehicle_horizontal_axes must contain two distinct indices from [0, 1, 2].")
    camera_to_vehicle_rotation = camera_to_vehicle[:, :3, :3]
    gram = camera_to_vehicle_rotation.transpose(1, 2) @ camera_to_vehicle_rotation
    determinant = torch.linalg.det(camera_to_vehicle_rotation)
    extrinsic_valid = torch.isfinite(camera_to_vehicle_rotation).all(dim=-1).all(dim=-1)
    extrinsic_valid &= (gram - torch.eye(3, device=gram.device, dtype=gram.dtype)).abs().amax(dim=(-1, -2)) < 0.05
    extrinsic_valid &= (determinant > 0.95) & (determinant < 1.05)
    if not bool(extrinsic_valid.all().item()):
        raise ValueError("camera_to_vehicle must contain a finite proper, approximately orthonormal calibration rotation.")
    batch_size, num_queries, _ = query_uv.shape
    if satellite_is_ego_oriented.any():
        forward_norm = torch.linalg.vector_norm(satellite_forward_world_xy, dim=-1)
        right_norm = torch.linalg.vector_norm(satellite_right_world_xy, dim=-1)
        axis_dot = (satellite_forward_world_xy * satellite_right_world_xy).sum(dim=-1).abs()
        invalid_axes = satellite_is_ego_oriented & (
            ~torch.isfinite(satellite_forward_world_xy).all(dim=-1)
            | ~torch.isfinite(satellite_right_world_xy).all(dim=-1)
            | (forward_norm < 1e-6)
            | (right_norm < 1e-6)
            | (axis_dot > 0.1)
        )
        if invalid_axes.any():
            raise ValueError(
                "Explicit satellite_forward_world_xy and satellite_right_world_xy must be finite, nonzero, "
                "and nearly orthogonal."
            )
    image_h, image_w = depth.shape[-2:]
    front_h, front_w = front_image_hw
    # Depth maps are aligned to the front image, but can be stored at another
    # resolution. Preserve pixel-center alignment when sampling them.
    scaled_uv = query_uv.clone()
    scaled_uv[..., 0] = query_uv[..., 0] * max(image_w - 1, 1) / max(front_w - 1, 1)
    scaled_uv[..., 1] = query_uv[..., 1] * max(image_h - 1, 1) / max(front_h - 1, 1)
    gx = scaled_uv[..., 0] / max(image_w - 1, 1) * 2.0 - 1.0
    gy = scaled_uv[..., 1] / max(image_h - 1, 1) * 2.0 - 1.0
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(2)
    valid_weight = F.grid_sample(depth_valid[:, None].float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0, :, 0]
    depth_weighted = F.grid_sample(
        (depth * depth_valid.to(depth.dtype))[:, None], grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )[:, 0, :, 0]
    sampled = depth_weighted / valid_weight.clamp_min(1e-6)
    if not 0.0 <= depth_valid_weight_threshold <= 1.0:
        raise ValueError("depth_valid_weight_threshold must lie in [0, 1].")
    if not 0.0 <= min_depth_confidence <= 1.0:
        raise ValueError("min_depth_confidence must lie in [0, 1].")
    valid = (valid_weight >= depth_valid_weight_threshold) & torch.isfinite(sampled) & (sampled > 0)
    if depth_confidence is not None:
        confidence = F.grid_sample(
            depth_confidence[:, None].float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        )[:, 0, :, 0].clamp(0.0, 1.0)
    else:
        confidence = torch.ones_like(sampled)
    valid &= confidence > min_depth_confidence

    fx = intrinsics[:, 0, 0:1]
    fy = intrinsics[:, 1, 1:2]
    cx = intrinsics[:, 0, 2:3]
    cy = intrinsics[:, 1, 2:3]
    ray = torch.stack(
        ((query_uv[..., 0] - cx) / fx.clamp_min(1e-8), (query_uv[..., 1] - cy) / fy.clamp_min(1e-8), torch.ones_like(sampled)),
        dim=-1,
    )
    ray_norm = torch.linalg.vector_norm(ray, dim=-1).clamp_min(1e-8)
    if depth_type.startswith("inverse_"):
        sampled = 1.0 / sampled.clamp_min(1e-8)
    if depth_type.endswith("ray_distance"):
        camera_z = sampled / ray_norm
    else:
        camera_z = sampled
    camera_point = ray * camera_z[..., None]
    rotation = camera_to_vehicle_rotation
    vehicle_vector = torch.matmul(camera_point, rotation.transpose(1, 2))
    axis_indices = torch.tensor(vehicle_horizontal_axes, device=vehicle_vector.device, dtype=torch.long)
    front_xy = vehicle_vector.index_select(-1, axis_indices)
    valid &= torch.isfinite(front_xy).all(dim=-1)
    front_xy = torch.where(valid[..., None], front_xy, torch.zeros_like(front_xy))
    reliability = confidence * valid.to(confidence.dtype)

    sat_h, sat_w = satellite_image_hw
    du = satellite_grid_uv[None, :, 0] - (sat_w - 1) * 0.5
    dv = satellite_grid_uv[None, :, 1] - (sat_h - 1) * 0.5
    mpp = satellite_meters_per_pixel[:, None]
    center = satellite_center_world_xy[:, None, :]
    north_up_xy = torch.stack((center[..., 0] + du * mpp, center[..., 1] - dv * mpp), dim=-1)
    forward = satellite_forward_world_xy[:, None, :]
    right = satellite_right_world_xy[:, None, :]
    ego_xy = center + (du * mpp)[..., None] * right - (dv * mpp)[..., None] * forward
    is_ego = satellite_is_ego_oriented[:, None, None]
    satellite_xy = torch.where(is_ego, ego_xy, north_up_xy)
    return CandidateGeometry(
        front_xy=front_xy,
        front_valid=valid,
        front_reliability=reliability,
        satellite_xy=satellite_xy,
    )


def satellite_uv_to_world_xy(
    uv: torch.Tensor,
    meters_per_pixel: torch.Tensor,
    center_world_xy: torch.Tensor,
    forward_world_xy: torch.Tensor,
    right_world_xy: torch.Tensor,
    is_ego_oriented: torch.Tensor,
    satellite_image_hw: tuple[int, int],
) -> torch.Tensor:
    """Map satellite-image UV coordinates to world XY using explicit map axes."""

    sat_h, sat_w = satellite_image_hw
    du = uv[..., 0] - (sat_w - 1) * 0.5
    dv = uv[..., 1] - (sat_h - 1) * 0.5
    mpp = meters_per_pixel[:, None]
    center = center_world_xy[:, None, :]
    north_up = torch.stack((center[..., 0] + du * mpp, center[..., 1] - dv * mpp), dim=-1)
    ego = center + (du * mpp)[..., None] * right_world_xy[:, None, :] - (dv * mpp)[..., None] * forward_world_xy[:, None, :]
    return torch.where(is_ego_oriented[:, None, None], ego, north_up)
