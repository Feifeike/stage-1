from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1.evaluation.anchor_scale import estimate_ransac_scale, weighted_mean_scale


class AnchorScaleRansacTest(unittest.TestCase):
    def test_no_outliers(self) -> None:
        self.assertAlmostEqual(weighted_mean_scale([1.0, 3.0], [1.0, 3.0]), 2.5)
        result = estimate_ransac_scale(
            [20.0, 40.0, 60.0, 80.0],
            [10.0, 20.0, 30.0, 40.0],
            weights=[1.0, 2.0, 3.0, 4.0],
        )
        self.assertEqual(result.status, "ransac")
        self.assertAlmostEqual(result.scale, 2.0)
        self.assertEqual(result.final_inlier_count, 4)
        self.assertAlmostEqual(result.inlier_ratio, 1.0)

    def test_with_outlier(self) -> None:
        result = estimate_ransac_scale(
            [20.0, 20.0, 21.0, 19.0, 100.0],
            [10.0, 10.0, 10.0, 10.0, 10.0],
        )
        self.assertEqual(result.status, "ransac")
        self.assertAlmostEqual(result.scale, 2.0)
        self.assertEqual(result.final_inlier_count, 4)
        self.assertAlmostEqual(result.inlier_ratio, 0.8)
        self.assertEqual(result.inlier_mask, (True, True, True, True, False))

    def test_insufficient_anchors_can_fallback_to_weighted_mean_or_none(self) -> None:
        weighted_result = estimate_ransac_scale(
            [20.0, 40.0],
            [10.0, 20.0],
            weights=[1.0, 3.0],
            min_anchors=3,
            fallback="weighted_mean",
        )
        self.assertEqual(weighted_result.status, "fallback_weighted_mean_insufficient_anchors")
        self.assertAlmostEqual(weighted_result.scale, 2.0)

        none_result = estimate_ransac_scale(
            [20.0, 40.0],
            [10.0, 20.0],
            min_anchors=3,
            fallback="none",
        )
        self.assertEqual(none_result.status, "no_correction_insufficient_anchors")
        self.assertIsNone(none_result.scale)


if __name__ == "__main__":
    unittest.main()
