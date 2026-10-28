from __future__ import annotations

import unittest

import torch

from models.candidate_graph import (
    CandidateGeometry,
    build_candidate_geometry,
    estimate_shared_similarity,
    geometry_edge_features,
    select_diverse_candidates,
)
from models.matcher import CrossViewCoarseMatcher


class CandidateGraphTests(unittest.TestCase):
    def _matcher(self, enabled: bool = True) -> CrossViewCoarseMatcher:
        return CrossViewCoarseMatcher(
            dim=8,
            num_layers=1,
            num_heads=2,
            temperature=1.0,
            candidate_graph_config={
                "enabled": enabled,
                "candidate_count": 3,
                "candidate_min_separation_px": 0.0,
                "hidden_dim": 16,
                "num_layers": 1,
                "use_geometry": True,
                "depth_scale_mode": "robust",
                "max_logit_residual": 20.0,
            },
        )

    @staticmethod
    def _geometry(valid: bool = True) -> CandidateGeometry:
        return CandidateGeometry(
            front_xy=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]]]),
            front_valid=torch.full((1, 6), valid, dtype=torch.bool),
            front_reliability=torch.ones((1, 6)),
            satellite_xy=torch.tensor(
                [[[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0], [6.0, 0.0], [6.0, 3.0], [9.0, 0.0], [9.0, 3.0]]]
            ),
        )

    def test_geometry_prefers_consistent_direction_and_fits_global_scale(self) -> None:
        front = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        front_valid = torch.ones((2,), dtype=torch.bool)
        reliability = torch.ones((2,))
        valid_candidates = torch.ones((2, 1), dtype=torch.bool)
        src = torch.tensor([0])
        dst = torch.tensor([1])
        good = torch.tensor([[[0.0, 0.0]], [[3.0, 0.0]]])
        reversed_direction = torch.tensor([[[0.0, 0.0]], [[-3.0, 0.0]]])
        good_features, good_compatibility, _ = geometry_edge_features(
            front, front_valid, reliability, good, valid_candidates, src, dst, 1, shared_scale=3.0
        )
        bad_features, bad_compatibility, _ = geometry_edge_features(
            front, front_valid, reliability, reversed_direction, valid_candidates, src, dst, 1, shared_scale=3.0
        )
        self.assertGreater(float(good_compatibility.item()), float(bad_compatibility.item()))
        self.assertLess(float(good_features[0, 3]), float(bad_features[0, 3]))

        scaled = torch.tensor([[[0.0, 0.0]], [[15.0, 0.0]]])
        scale = torch.tensor(15.0)
        _, scale_compatibility, _ = geometry_edge_features(
            front, front_valid, reliability, scaled, valid_candidates, src, dst, 1, shared_scale=scale
        )
        self.assertAlmostEqual(float(good_compatibility.item()), float(scale_compatibility.item()), places=5)

    def test_candidate_selection_keeps_separated_position_peaks(self) -> None:
        grid = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
             [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
             [0.0, 2.0], [1.0, 2.0], [2.0, 2.0]]
        )
        scores = torch.tensor([[[10.0, 9.0, 1.0, 8.0, 0.0, 0.0, 0.0, 0.0, 7.0]]])
        indices, valid = select_diverse_candidates(
            scores,
            torch.ones_like(scores, dtype=torch.bool),
            grid,
            grid_hw=(3, 3),
            count=3,
            min_separation_px=1.1,
        )
        self.assertTrue(valid.all())
        selected_uv = grid[indices[0, 0]]
        distances = torch.cdist(selected_uv[None], selected_uv[None]).squeeze(0)
        self.assertGreaterEqual(float(distances[0, 1]), 1.1)
        self.assertGreaterEqual(float(distances[0, 2]), 1.1)
        self.assertGreaterEqual(float(distances[1, 2]), 1.1)

    def test_shared_transform_estimation_uses_one_proper_rotation_and_scale(self) -> None:
        front = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]]
        )
        theta = torch.tensor(0.6)
        rotation_gt = torch.tensor(
            [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]
        )
        q = 2.5 * (front @ rotation_gt.T) + torch.tensor([7.0, -4.0])
        pairs = torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [4, 5], [0, 5]])
        scale, rotation, valid = estimate_shared_similarity(
            front,
            torch.ones((6,), dtype=torch.bool),
            q,
            torch.ones((6,), dtype=torch.bool),
            pairs,
            mode="robust",
        )
        self.assertTrue(valid)
        self.assertAlmostEqual(float(scale.item()), 2.5, places=4)
        torch.testing.assert_close(rotation, rotation_gt, atol=1e-4, rtol=1e-4)
        self.assertGreater(float(torch.linalg.det(rotation)), 0.99)

    def test_candidate_logits_write_back_and_receive_gradients(self) -> None:
        torch.manual_seed(3)
        matcher = self._matcher(True)
        # Equal visual scores make the candidate graph's relative correction
        # solely responsible for selecting among the candidate set.
        front = torch.zeros((1, 6, 8), requires_grad=True)
        sat = torch.randn((1, 8, 8), requires_grad=True)
        grid = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
             [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
        )
        output = matcher(
            front,
            sat,
            grid,
            query_grid_hw=(2, 3),
            satellite_grid_hw=(2, 4),
            candidate_geometry=self._geometry(),
        )
        self.assertIsNotNone(output.candidate_logits)
        self.assertFalse(torch.allclose(output.pre_graph_logits, output.logits))
        self.assertTrue(torch.equal(output.coarse_indices, output.logits.argmax(dim=-1)))
        self.assertTrue(torch.any(output.coarse_indices != output.pre_graph_logits.argmax(dim=-1)))
        loss = torch.nn.functional.cross_entropy(output.logits.reshape(-1, 8), torch.zeros((6,), dtype=torch.long))
        loss.backward()
        graph_parameters = list(matcher.candidate_graph.parameters())
        self.assertTrue(all(parameter.grad is not None for parameter in graph_parameters))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in graph_parameters))
        grad_norm = sum(parameter.grad.detach().abs().sum() for parameter in graph_parameters)
        self.assertTrue(torch.isfinite(grad_norm))
        self.assertGreater(float(grad_norm), 0.0)

    def test_no_valid_depth_keeps_finite_matching(self) -> None:
        matcher = self._matcher(True)
        front = torch.randn((1, 6, 8))
        sat = torch.randn((1, 8, 8))
        grid = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
             [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
        )
        output = matcher(
            front,
            sat,
            grid,
            query_grid_hw=(2, 3),
            satellite_grid_hw=(2, 4),
            candidate_geometry=self._geometry(False),
        )
        self.assertTrue(torch.isfinite(output.probabilities).all())
        self.assertTrue(torch.isfinite(output.coarse_uv[output.coarse_valid_mask]).all())
        self.assertEqual(float(output.graph_edge_valid_fraction.item()), 0.0)

    def test_disabled_graph_preserves_raw_matcher_path(self) -> None:
        torch.manual_seed(8)
        enabled_model = self._matcher(False)
        front = torch.randn((1, 6, 8))
        sat = torch.randn((1, 8, 8))
        grid = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
             [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
        )
        output = enabled_model(front, sat, grid)
        base = torch.nn.functional.normalize(output.contextual_front, dim=-1) @ torch.nn.functional.normalize(
            output.contextual_satellite, dim=-1
        ).transpose(1, 2)
        base = base * enabled_model.logit_scale.exp().clamp(1.0, 100.0)
        torch.testing.assert_close(output.logits, base)
        self.assertIsNone(output.candidate_indices)

    def test_empty_depth_mask_produces_explicit_invalid_geometry(self) -> None:
        uv = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
        sat_uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        geometry = build_candidate_geometry(
            query_uv=uv,
            satellite_grid_uv=sat_uv,
            depth=torch.zeros((1, 4, 4)),
            depth_valid=torch.zeros((1, 4, 4), dtype=torch.bool),
            depth_confidence=torch.zeros((1, 4, 4)),
            intrinsics=torch.tensor([[[1.0, 0.0, 1.5], [0.0, 1.0, 1.5], [0.0, 0.0, 1.0]]]),
            camera_to_vehicle=torch.eye(4)[None],
            vehicle_horizontal_axes=(0, 2),
            satellite_meters_per_pixel=torch.tensor([1.0]),
            satellite_center_world_xy=torch.zeros((1, 2)),
            satellite_forward_world_xy=torch.tensor([[0.0, 1.0]]),
            satellite_right_world_xy=torch.tensor([[1.0, 0.0]]),
            satellite_is_ego_oriented=torch.tensor([False]),
            front_image_hw=(4, 4),
            satellite_image_hw=(2, 2),
            depth_type="camera_z",
        )
        self.assertFalse(geometry.front_valid.any())
        self.assertTrue(torch.isfinite(geometry.satellite_xy).all())


if __name__ == "__main__":
    unittest.main()
