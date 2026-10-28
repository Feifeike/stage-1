from __future__ import annotations

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1.models.matcher import CrossViewCoarseMatcher, compute_dual_softmax_log_scores


def test_dual_softmax_is_batch_local_and_masks_invalid_tokens() -> None:
    logits = torch.tensor(
        [
            [[2.0, 0.0, -1.0, 99.0], [0.0, 3.0, 1.0, 99.0], [99.0, 99.0, 99.0, 99.0]],
            [[-100.0, 500.0, 4.0, 2.0], [99.0, 99.0, 99.0, 99.0], [1.0, 99.0, 2.0, 0.0]],
        ],
        dtype=torch.float16,
    )
    query_valid = torch.tensor([[True, True, False], [True, False, True]])
    satellite_valid = torch.tensor([[True, True, True, False], [True, False, True, True]])

    scores = compute_dual_softmax_log_scores(logits, query_valid, satellite_valid)
    assert scores.dtype == torch.float32
    assert torch.isneginf(scores[0, 2]).all()
    assert torch.isneginf(scores[0, :, 3]).all()
    assert torch.isneginf(scores[1, 1]).all()
    assert torch.isneginf(scores[1, :, 1]).all()

    changed_other_sample = logits.clone()
    changed_other_sample[1] *= 1000
    scores_after = compute_dual_softmax_log_scores(changed_other_sample, query_valid, satellite_valid)
    torch.testing.assert_close(scores[0], scores_after[0])


def test_all_invalid_match_has_no_nan_or_pseudo_match() -> None:
    matcher = CrossViewCoarseMatcher(dim=3, num_layers=0, num_heads=1, temperature=1.0, score_mode="dual_softmax")
    matcher.eval()
    front = torch.ones((1, 2, 3))
    satellite = torch.ones((1, 2, 3))
    grid = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    output = matcher(
        front,
        satellite,
        grid,
        query_valid_mask=torch.zeros((1, 2), dtype=torch.bool),
        satellite_valid_mask=torch.ones((1, 2), dtype=torch.bool),
    )

    assert output.coarse_indices.tolist() == [[-1, -1]]
    assert torch.isnan(output.coarse_uv).all()
    assert torch.equal(output.probabilities, torch.zeros_like(output.probabilities))
    assert torch.equal(output.matching_scores, torch.zeros_like(output.matching_scores))
    assert torch.isneginf(output.dual_log_scores).all()
    assert torch.isfinite(output.matchability_logits).all()

    no_candidates = matcher(
        front,
        satellite,
        grid,
        query_valid_mask=torch.ones((1, 2), dtype=torch.bool),
        satellite_valid_mask=torch.zeros((1, 2), dtype=torch.bool),
    )
    assert no_candidates.coarse_indices.tolist() == [[-1, -1]]
    assert torch.isnan(no_candidates.coarse_uv).all()
    assert torch.equal(no_candidates.matching_scores, torch.zeros_like(no_candidates.matching_scores))
    assert torch.isneginf(no_candidates.dual_log_scores).all()

    one_candidate = matcher(
        front,
        satellite,
        grid,
        query_valid_mask=torch.ones((1, 2), dtype=torch.bool),
        satellite_valid_mask=torch.tensor([[True, False]]),
    )
    assert torch.equal(one_candidate.probabilities[..., 1], torch.zeros_like(one_candidate.probabilities[..., 1]))
    torch.testing.assert_close(one_candidate.probabilities[..., 0], torch.ones_like(one_candidate.probabilities[..., 0]))
    assert one_candidate.coarse_indices.tolist() == [[0, 0]]


def test_dual_selection_is_inference_only_and_default_is_unchanged() -> None:
    matcher = CrossViewCoarseMatcher(dim=3, num_layers=0, num_heads=1, temperature=1.0, score_mode="dual_softmax")
    front = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    satellite = torch.tensor([[[0.9, (0.19) ** 0.5, 0.0], [0.8, 0.0, 0.6]]])
    grid = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

    matcher.train()
    training_output = matcher(front, satellite, grid)
    assert training_output.score_mode == "forward_softmax"
    assert training_output.coarse_indices.tolist()[0][0] == 0
    torch.testing.assert_close(training_output.probabilities, torch.softmax(training_output.logits, dim=-1))

    matcher.eval()
    inference_output = matcher(front, satellite, grid)
    assert inference_output.score_mode == "dual_softmax"
    assert inference_output.coarse_indices.tolist()[0][0] == 1
    torch.testing.assert_close(inference_output.matching_scores, inference_output.dual_log_scores.exp())
    torch.testing.assert_close(inference_output.score_dual, inference_output.dual_log_scores.exp())
    assert not torch.allclose(
        inference_output.matching_scores.sum(dim=-1),
        torch.ones_like(inference_output.matching_scores.sum(dim=-1)),
    )

    default_matcher = CrossViewCoarseMatcher(dim=3, num_layers=0, num_heads=1, temperature=1.0)
    default_matcher.eval()
    default_output = default_matcher(front, satellite, grid)
    assert default_output.score_mode == "forward_softmax"
    assert default_output.score_dual is None
    assert torch.equal(default_output.coarse_indices, default_output.logits.argmax(dim=-1))
    torch.testing.assert_close(default_output.probabilities, torch.softmax(default_output.logits, dim=-1))
