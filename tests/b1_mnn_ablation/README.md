# B1 MNN matching ablation

This adds a mutual nearest neighbor (MNN) selector to the existing forward-softmax and dual-softmax comparison. All three modes share the same temperature-scaled logits from one forward pass, checkpoint, KITTI-360 validation manifest, preprocessing, and GT-valid query set. No weights were retrained.

## Setup

- Stage-1 checkpoint: `outputs_0923/stage1_refine/checkpoints/stage1_latest.pth` (epoch 91, step 613795; SHA-256 `171c2636c931be1c8d025439624fa877267f0d501dcb46d345232dc4cd96793d`)
- Confidence checkpoint: `outputs_1021/stage1_confidence/checkpoints/confidence_best.pth` (epoch 6, step 40470; SHA-256 `495685abf2e17cc74cb10fc058619ea12b2df49d2fdc68d51cfb0d6eb65f5f66`)
- Data: 400-frame KITTI-360 validation manifest, 50 batches, 484,771 valid-GT queries, 0 load errors
- Device: CPU; elapsed time: 2,136 seconds
- Coarse correctness threshold: geometric error `<= 10 px`

MNN accepts a pair only when the satellite is the front query's row argmax and that front query is also the satellite's column argmax, within the same image. Invalid tokens are excluded, there is no dustbin, and non-mutual queries have no MNN prediction. The matching selector uses the already scaled logits without extra temperature processing.

## Coarse matching and position results

| Mode | All-query Top-1 <= 10 px | Accepted-query coverage | Precision among accepted pairs | Coarse position error mean / median |
|---|---:|---:|---:|---:|
| Forward softmax | 56.20% | 100.00% | 56.20% | 15.54 / 9.06 px |
| Dual softmax | 49.40% | 100.00% | 49.40% | 20.00 / 10.11 px |
| MNN | 1.01% | 2.11% (10,239 / 484,771) | 47.58% | 16.50 / 10.53 px among accepted pairs |

MNN's all-query Top-1 counts the 474,532 non-mutual valid-GT queries as incorrect. Its accepted-pair precision is conditional on the 10,239 MNN matches and should be read together with the 2.11% coverage. On this validation set, MNN has lower conditional precision than both dense selectors and very low coverage.

Refined position error is 12.04 / 4.67 px for forward softmax, 15.06 / 5.75 px for dual softmax, and 292.94 / 295.09 px for MNN. The MNN figure covers only its 10,239 accepted queries. The existing refiner operates on a dense query grid; MNN leaves most positions unmatched, so the unchanged refiner receives zero-filled coarse-warp context at those positions and their outputs are masked. Treat this MNN refined result as a diagnostic of applying the current dense refiner to sparse MNN matches, rather than a standalone MNN coarse-matcher score.

## Equal-ratio retention

The forward and dual rows rank all 484,771 valid-GT queries. MNN ranks only its 10,239 accepted pairs, so its retained counts differ; its coverage column always uses all 484,771 queries as the denominator.

| Retained fraction within each mode's ranking population | Forward precision / all-query coverage | Dual precision / all-query coverage | MNN precision / all-query coverage |
|---|---:|---:|---:|
| 25% | 85.07% / 21.27% | 54.82% / 13.70% | 77.03% / 0.41% (2,560 retained) |
| 50% | 75.44% / 37.72% | 54.74% / 27.37% | 66.45% / 0.70% (5,120 retained) |
| 75% | 66.44% / 49.83% | 53.30% / 39.98% | 57.21% / 0.91% (7,680 retained) |

MNN retention is ranked by the accepted pair's original forward-softmax score. This is a ranking score, not a calibrated correctness probability. Since MNN's ranking population is sparse, its retained precision/coverage is not a same-count comparison with forward or dual softmax.

## Depth metrics

The uncorrected geometric camera-z diagnostic uses the selected satellite position and camera pose, compared with LiDAR camera-z at the front query. Columns are AbsRel / SqRel / RMSE (m) / delta1. MNN metrics cover accepted matches only.

| Mode | Coarse geometric camera-z | Refined geometric camera-z |
|---|---:|---:|
| Forward softmax | 0.1731 / 0.6260 / 3.405 / 0.7294 | 0.1117 / 0.4768 / 2.958 / 0.8787 |
| Dual softmax | 0.2705 / 2.6204 / 5.127 / 0.6670 | 0.1773 / 1.2710 / 3.773 / 0.8223 |
| MNN | 0.1146 / 0.4324 / 3.201 / 0.8837 | 2.2353 / 78.9707 / 29.452 / 0.0907 |

The project-standard confidence-weighted scale correction uses the same frozen confidence head for all modes. Coarse summaries use the 397 frames with a usable scale for all three modes. Refined summaries use the 375 paired frames with a usable scale for all three modes; 25 frames lack a usable MNN-refined scale. Each metric below is the mean of per-image metrics on its paired frame set.

| Stage / mode | AbsRel | SqRel | RMSE (m) | delta1 |
|---|---:|---:|---:|---:|
| Coarse / forward | 0.10479 | 0.62674 | 3.61733 | 0.90176 |
| Coarse / dual | 0.11355 | 0.64337 | 3.63420 | 0.90143 |
| Coarse / MNN | 0.10784 | 0.63149 | 3.56767 | 0.90361 |
| Refined / forward | 0.10442 | 0.72096 | 3.73169 | 0.89463 |
| Refined / dual | 0.10905 | 0.67952 | 3.64887 | 0.90048 |
| Refined / MNN | 1.68046 | 52.85383 | 25.64865 | 0.11401 |

The refined confidence-weighted MNN metric is poor and uses only 7,976 valid anchors in the paired 375 frames (about 21 per frame), versus roughly 1,940–1,947 per frame for the dense modes. It reflects the sparse MNN anchors and dense-refiner behavior; it is not evidence that the confidence head was retrained or recalibrated for MNN.

## Files and reproduction

- Full metrics and provenance: [`metrics.json`](metrics.json)
- Batch log: [`../b1_mnn_ablation.log`](../b1_mnn_ablation.log)
- Evaluator: [`scripts/evaluate_matching_ab.py`](../../scripts/evaluate_matching_ab.py)

```bash
conda activate bevsplat
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/evaluate_matching_ab.py \
  --config configs/config_confidence.yaml \
  --checkpoint outputs_0923/stage1_refine/checkpoints/stage1_latest.pth \
  --confidence_checkpoint outputs_1021/stage1_confidence/checkpoints/confidence_best.pth \
  --manifest outputs_0923/stage1_refine/kitti360_val400_manifest.json \
  --output_dir outputs_1026/b1_mnn_ablation \
  --geometry_threshold_px 10 \
  --device cpu --batch_size 8 --num_workers 0
```
