from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbone import Stage1Backbone
from ..data.geometry import Stage1Targets, build_stage1_targets, make_query_grid, make_satellite_grid
from .candidate_graph import (
    CandidateGeometry,
    build_candidate_geometry,
    satellite_uv_to_world_xy,
)
from .matcher import CrossViewCoarseMatcher, MatcherOutput
from .refiner import LocalRefiner


@dataclass
class Stage1Output:
    front_features: torch.Tensor
    satellite_features: torch.Tensor
    front_tokens: torch.Tensor
    satellite_tokens: torch.Tensor
    satellite_grid_uv: torch.Tensor
    targets: Stage1Targets
    matcher: MatcherOutput
    refined_uv: torch.Tensor
    certainty_logits: torch.Tensor
    certainty: torch.Tensor
    satellite_meters_per_pixel: torch.Tensor
    refined_world_xy: torch.Tensor
    satellite_geometry_valid: torch.Tensor


class Stage1CorrespondenceModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        matcher_cfg = config["matcher"]
        matching_cfg = config.get("matching", {})
        refiner_cfg = config["refiner"]
        self.backbone = Stage1Backbone(config)
        dim = int(model_cfg.get("descriptor_dim", 128))
        self.matcher = CrossViewCoarseMatcher(
            dim=dim,
            num_layers=int(matcher_cfg.get("num_layers", 2)),
            num_heads=int(matcher_cfg.get("num_heads", 4)),
            temperature=float(matcher_cfg.get("temperature", 0.07)),
            score_mode=str(matching_cfg.get("score_mode", "forward_softmax")),
            candidate_graph_config=matcher_cfg.get("candidate_graph", {}),
        )
        self.candidate_graph_cfg = matcher_cfg.get("candidate_graph", {})
        if self.matcher.candidate_graph_enabled and bool(self.candidate_graph_cfg.get("use_geometry", True)):
            if not self.candidate_graph_cfg.get("depth_type"):
                raise ValueError(
                    "matcher.candidate_graph.depth_type must explicitly name the monocular depth definition "
                    "(camera_z, ray_distance, inverse_camera_z, or inverse_ray_distance)."
                )
            if not self.candidate_graph_cfg.get("depth_units"):
                raise ValueError(
                    "matcher.candidate_graph.depth_units must explicitly be 'metric' or 'arbitrary'."
                )
            axes = self.candidate_graph_cfg.get("vehicle_horizontal_axes")
            if not isinstance(axes, (list, tuple)) or len(axes) != 2:
                raise ValueError(
                    "candidate graph geometry requires explicit vehicle_horizontal_axes [right, forward] indices."
                )
            map_orientation = str(self.candidate_graph_cfg.get("satellite_orientation_default", "unknown"))
            if map_orientation not in {"north_up", "unknown"}:
                raise ValueError("satellite_orientation_default must be 'north_up' or 'unknown'.")
            source = str(self.candidate_graph_cfg.get("depth_source", "mono"))
            if source != "mono":
                raise ValueError("candidate graph geometry currently accepts depth_source='mono' only.")
        self.refiner = LocalRefiner(
            dim=dim,
            radius=int(refiner_cfg.get("radius", 2)),
            hidden_dim=(
                int(refiner_cfg["hidden_dim"])
                if refiner_cfg.get("hidden_dim") is not None
                else None
            ),
            max_offset_px=float(refiner_cfg.get("max_offset_px", 16.0)),
            scales=tuple(int(scale) for scale in refiner_cfg.get("scales", [16, 8, 4, 2, 1])),
            fine_feature_dim=int(refiner_cfg.get("fine_feature_dim", 64)),
            fine_pretrained=bool(refiner_cfg.get("fine_pretrained", False)),
            hidden_blocks=int(refiner_cfg.get("hidden_blocks", 8)),
            depthwise=bool(refiner_cfg.get("depthwise", True)),
            displacement_emb_dims={
                int(key): int(value)
                for key, value in refiner_cfg.get(
                    "displacement_emb_dims",
                    {16: 128, 8: 64, 4: 32, 2: 16, 1: 6},
                ).items()
            },
            local_corr_radii={
                int(key): (None if value is None else int(value))
                for key, value in refiner_cfg.get(
                    "local_corr_radii",
                    {16: 7, 8: 3, 4: 2, 2: None, 1: None},
                ).items()
            },
        )
        self.lidar_query_radius_px = int(config.get("geometry", {}).get("lidar_query_radius_px", 0))
        self.refiner_enabled = True
        self.coarse_no_grad = False

    def set_training_stage(self, stage: str) -> None:
        """Configure the existing two-stage coarse/refine training behavior."""

        if stage not in {"coarse", "refine", "joint"}:
            raise ValueError("training.stage must be one of: coarse, refine, joint.")
        self.refiner_enabled = stage != "coarse"
        self.coarse_no_grad = stage == "refine"

    def load_compatible_checkpoint(self, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
        """Load old Stage-1 weights while allowing only new graph parameters missing.

        All unrelated missing or unexpected keys remain errors, so adding the
        optional module cannot hide checkpoint/model incompatibilities.
        """

        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_prefixes = ("matcher.candidate_graph.",)
        unexpected_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_prefixes)
        ]
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Incompatible Stage-1 checkpoint: "
                f"missing={unexpected_missing}, unexpected={incompatible.unexpected_keys}"
            )
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)

    @staticmethod
    def _flatten(feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map.flatten(2).transpose(1, 2).contiguous()

    def forward(self, batch) -> Stage1Output:
        coarse_grad_enabled = torch.is_grad_enabled() and not self.coarse_no_grad
        with torch.set_grad_enabled(coarse_grad_enabled):
            front_features, satellite_features = self.backbone(batch.front_rgb, batch.satellite_rgb)
        batch_size = front_features.shape[0]
        front_image_hw = (batch.front_rgb.shape[-2], batch.front_rgb.shape[-1])
        satellite_image_hw = (batch.satellite_rgb.shape[-2], batch.satellite_rgb.shape[-1])
        query_uv = make_query_grid(
            batch_size=batch_size,
            feat_hw=front_features.shape[-2:],
            image_hw=front_image_hw,
            device=front_features.device,
            dtype=front_features.dtype,
        )
        satellite_grid_uv = make_satellite_grid(
            feat_hw=satellite_features.shape[-2:],
            image_hw=satellite_image_hw,
            device=satellite_features.device,
            dtype=satellite_features.dtype,
        )
        targets = build_stage1_targets(
            lidar_depth=batch.lidar_depth,
            query_uv=query_uv,
            intrinsics=batch.intrinsics,
            camera_to_world=batch.camera_to_world,
            satellite_meters_per_pixel=batch.satellite_meters_per_pixel,
            satellite_center_world_xy=batch.satellite_center_world_xy,
            satellite_image_hw=satellite_image_hw,
            explicit_negative_mask=batch.explicit_negative_mask if getattr(batch, "negative_available", None) is not None else None,
            lidar_query_radius_px=self.lidar_query_radius_px,
        )
        with torch.set_grad_enabled(coarse_grad_enabled):
            front_tokens = self._flatten(front_features)
            satellite_tokens = self._flatten(satellite_features)
            front_h, front_w = front_image_hw
            satellite_h, satellite_w = satellite_image_hw
            query_valid_mask = (
                torch.isfinite(query_uv).all(dim=-1)
                & (query_uv[..., 0] >= 0)
                & (query_uv[..., 0] <= front_w - 1)
                & (query_uv[..., 1] >= 0)
                & (query_uv[..., 1] <= front_h - 1)
            )
            satellite_valid_mask = (
                torch.isfinite(satellite_grid_uv).all(dim=-1)
                & (satellite_grid_uv[:, 0] >= 0)
                & (satellite_grid_uv[:, 0] <= satellite_w - 1)
                & (satellite_grid_uv[:, 1] >= 0)
                & (satellite_grid_uv[:, 1] <= satellite_h - 1)
            )
            satellite_valid_mask = satellite_valid_mask.unsqueeze(0).expand(batch_size, -1)
            candidate_geometry = None
            if self.matcher.candidate_graph_enabled:
                depth = getattr(batch, "da3_depth", None)
                depth_valid = getattr(batch, "da3_depth_valid_mask", None)
                depth_confidence = getattr(batch, "da3_depth_confidence", None)
                if depth is None:
                    depth = batch.front_rgb.new_zeros((batch_size, *front_image_hw))
                if depth_valid is None:
                    depth_valid = torch.zeros_like(depth, dtype=torch.bool)
                if depth_confidence is None:
                    depth_confidence = depth_valid.to(depth.dtype)
                depth_type = str(self.candidate_graph_cfg.get("depth_type", "camera_z"))
                candidate_geometry = build_candidate_geometry(
                    query_uv=query_uv,
                    satellite_grid_uv=satellite_grid_uv,
                    depth=depth,
                    depth_valid=depth_valid,
                    depth_confidence=depth_confidence,
                    intrinsics=batch.intrinsics,
                    camera_to_vehicle=batch.camera_to_vehicle,
                    vehicle_horizontal_axes=tuple(int(axis) for axis in self.candidate_graph_cfg.get("vehicle_horizontal_axes", [1, 0])),
                    satellite_meters_per_pixel=batch.satellite_meters_per_pixel,
                    satellite_center_world_xy=batch.satellite_center_world_xy,
                    satellite_forward_world_xy=batch.satellite_forward_world_xy,
                    satellite_right_world_xy=batch.satellite_right_world_xy,
                    satellite_is_ego_oriented=batch.satellite_is_ego_oriented,
                    front_image_hw=front_image_hw,
                    satellite_image_hw=satellite_image_hw,
                    depth_type=depth_type,
                    depth_valid_weight_threshold=float(self.candidate_graph_cfg.get("depth_valid_weight_threshold", 0.5)),
                    min_depth_confidence=float(self.candidate_graph_cfg.get("min_depth_confidence", 0.0)),
                )
                map_geometry_valid = getattr(
                    batch,
                    "satellite_geometry_valid",
                    torch.zeros((batch_size,), device=front_features.device, dtype=torch.bool),
                ).to(device=front_features.device, dtype=torch.bool)
                if str(self.candidate_graph_cfg.get("satellite_orientation_default", "unknown")) != "north_up":
                    map_geometry_valid &= batch.satellite_is_ego_oriented
                candidate_geometry.front_valid &= map_geometry_valid[:, None]
                candidate_geometry.front_reliability *= map_geometry_valid[:, None].to(
                    candidate_geometry.front_reliability.dtype
                )
            matcher = self.matcher(
                front_tokens,
                satellite_tokens,
                satellite_grid_uv=satellite_grid_uv,
                query_valid_mask=query_valid_mask,
                satellite_valid_mask=satellite_valid_mask,
                query_grid_hw=tuple(int(v) for v in front_features.shape[-2:]),
                satellite_grid_hw=tuple(int(v) for v in satellite_features.shape[-2:]),
                candidate_geometry=candidate_geometry,
            )
        if self.refiner_enabled:
            refiner_satellite = satellite_features.detach() if self.coarse_no_grad else satellite_features
            refined_uv, delta_certainty = self.refine_from_coarse(
                batch=batch,
                output_features=(front_features.detach() if self.coarse_no_grad else front_features, refiner_satellite),
                front_query_uv=query_uv,
                coarse_uv=matcher.coarse_uv.detach(),
                match_valid_mask=matcher.coarse_valid_mask,
                initial_matchability_logits=matcher.matchability_logits,
            )
        else:
            refined_uv = matcher.coarse_uv.detach()
            delta_certainty = matcher.matchability_logits.new_zeros(matcher.matchability_logits.shape)
        certainty_logits = matcher.matchability_logits + delta_certainty
        refined_world_xy = satellite_uv_to_world_xy(
            refined_uv,
            batch.satellite_meters_per_pixel,
            batch.satellite_center_world_xy,
            batch.satellite_forward_world_xy,
            batch.satellite_right_world_xy,
            batch.satellite_is_ego_oriented,
            satellite_image_hw,
        )
        satellite_geometry_valid = getattr(
            batch,
            "satellite_geometry_valid",
            torch.zeros((batch_size,), device=front_features.device, dtype=torch.bool),
        ).to(device=front_features.device, dtype=torch.bool)
        if (
            self.matcher.candidate_graph_enabled
            and str(self.candidate_graph_cfg.get("satellite_orientation_default", "unknown")) != "north_up"
        ):
            satellite_geometry_valid &= batch.satellite_is_ego_oriented
        return Stage1Output(
            front_features=front_features,
            satellite_features=satellite_features,
            front_tokens=front_tokens,
            satellite_tokens=satellite_tokens,
            satellite_grid_uv=satellite_grid_uv,
            targets=targets,
            matcher=matcher,
            refined_uv=refined_uv,
            certainty_logits=certainty_logits,
            certainty=torch.sigmoid(certainty_logits),
            satellite_meters_per_pixel=batch.satellite_meters_per_pixel,
            refined_world_xy=refined_world_xy,
            satellite_geometry_valid=satellite_geometry_valid,
        )

    def refine_from_coarse(
        self,
        *,
        batch,
        output_features: tuple[torch.Tensor, torch.Tensor] | None = None,
        front_query_uv: torch.Tensor,
        coarse_uv: torch.Tensor,
        match_valid_mask: torch.Tensor,
        initial_matchability_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the existing refiner for externally selected coarse matches."""

        if not self.refiner_enabled:
            refined_uv = coarse_uv.masked_fill(~match_valid_mask[..., None], torch.nan)
            return refined_uv, initial_matchability_logits.new_zeros(initial_matchability_logits.shape)
        if output_features is None:
            front_features, satellite_features = self.backbone(batch.front_rgb, batch.satellite_rgb)
        else:
            front_features, satellite_features = output_features
        satellite_image_hw = (batch.satellite_rgb.shape[-2], batch.satellite_rgb.shape[-1])
        safe_coarse_uv = torch.where(match_valid_mask[..., None], coarse_uv, torch.zeros_like(coarse_uv))
        refined_uv, delta_certainty = self.refiner(
            front_features=front_features,
            satellite_feature_map=satellite_features,
            front_rgb=batch.front_rgb,
            satellite_rgb=batch.satellite_rgb,
            front_query_uv=front_query_uv,
            coarse_uv=safe_coarse_uv,
            satellite_image_hw=satellite_image_hw,
            initial_matchability_logits=initial_matchability_logits,
        )
        refined_uv = refined_uv.masked_fill(~match_valid_mask[..., None], torch.nan)
        delta_certainty = delta_certainty.masked_fill(~match_valid_mask, 0.0)
        return refined_uv, delta_certainty
