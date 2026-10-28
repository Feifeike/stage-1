# B1：forward 与 dual-softmax 对比

本次使用现有 confidence head 配套的 Stage-1 `stage1_latest.pth` 重新评估，并删除了先前基于 epoch 6 Stage-1 checkpoint 的 B1 结果。A/B 共用同一次前向计算得到的温度处理后 logits，只改变 coarse 匹配选择方式；refine 网络和细化流程保持不变。粗匹配 Top-1 按选中卫星位置与 GT 位置的几何误差 `<= 10 px` 计算。

后续新增的 MNN 消融及结果见 [`b1_mnn_ablation/README.md`](../b1_mnn_ablation/README.md)。

## 权重和数据

- Stage-1 checkpoint：`outputs_0923/stage1_refine/checkpoints/stage1_latest.pth`
- Stage-1 训练位置：epoch 91，step 613795
- Stage-1 SHA-256：`171c2636c931be1c8d025439624fa877267f0d501dcb46d345232dc4cd96793d`
- Confidence checkpoint：`outputs_1021/stage1_confidence/checkpoints/confidence_best.pth`
- Confidence head 训练位置：epoch 6，step 40470
- Confidence checkpoint SHA-256：`495685abf2e17cc74cb10fc058619ea12b2df49d2fdc68d51cfb0d6eb65f5f66`
- Confidence checkpoint 记录的底模路径：`models/stage1/outputs_0923/stage1_refine/checkpoints/stage1_latest.pth`
- 数据集：`outputs_0923/stage1_refine/kitti360_val400_manifest.json`，400 帧 KITTI-360 验证集
- 设备：CPU，batch size 8，50 batches，约 1524.45 秒
- 有效 GT 查询：484771；加载错误：0

Confidence checkpoint 是旧格式，没有保存底模 SHA-256；它保存的底模路径与本次指定路径相同，但无法从该旧 checkpoint 独立校验训练时底模文件的哈希。本次使用的权重文件哈希如上。

## 匹配结果

| 指标 | A：forward softmax | B：dual softmax |
|---|---:|---:|
| 粗匹配 Top-1，误差 ≤10 px | 56.20% | 49.40% |
| 粗匹配 exact-grid Top-1 | 51.73% | 45.22% |
| 粗匹配位置误差，均值 / 中位数 | 15.54 / 9.06 px | 20.00 / 10.11 px |
| Refine 位置误差，均值 / 中位数 | 12.04 / 4.67 px | 15.06 / 5.75 px |

保留比例按相同数量的有效 GT 查询全局排序。表中 precision 是保留查询中误差不超过 10 px 的比例，coverage 是正确查询数占全部有效 GT 查询的比例。

| 保留比例 | A：precision / coverage | B：precision / coverage |
|---|---:|---:|
| 25%（121193 查询） | 85.07% / 21.27% | 54.82% / 13.70% |
| 50%（242386 查询） | 75.44% / 37.72% | 54.74% / 27.37% |
| 75%（363579 查询） | 66.44% / 49.83% | 53.30% / 39.98% |

## Confidence scale 校正深度结果

对每种模式分别在其粗匹配位置和 refine 位置采样同一个冻结 confidence head 的权重，按项目主协议计算：

`scale = exp(sum_i normalized_confidence_i * log(anchor_z_i / mono_anchor_z_i))`

然后对稠密单目深度乘以每帧 scale，在相同有效 LiDAR mask 上计算指标。所有 inference-valid、有效锚点均参与 scale；GT mask 不用于选锚点或计算 confidence。主结果是 refine 阶段，使用全部 400 帧、10509909 个有效深度像素。粗匹配阶段是阶段消融：同一 confidence head 被应用到粗匹配位置；A/B 共同有有效 scale 的 397 帧、10432843 个像素。

| 阶段 / 指标 | A：forward softmax | B：dual softmax |
|---|---:|---:|
| Coarse：AbsRel | 0.10479 | 0.11355 |
| Coarse：SqRel | 0.62674 | 0.64337 |
| Coarse：RMSE (m) | 3.61733 | 3.63420 |
| Coarse：δ1 / δ2 / δ3 | 0.90176 / 0.95923 / 0.97493 | 0.90143 / 0.95988 / 0.97623 |
| Refine：AbsRel | 0.10336 | 0.10899 |
| Refine：SqRel | 0.69934 | 0.66695 |
| Refine：RMSE (m) | 3.68070 | 3.60847 |
| Refine：δ1 / δ2 / δ3 | 0.89791 / 0.95555 / 0.97116 | 0.90080 / 0.95637 / 0.97548 |

Refine 后 dual-softmax 的 AbsRel 从 0.10336 变为 0.10899，RMSE 从 3.68070 m 变为 3.60847 m；因此各深度指标方向不完全一致。匹配位置和 Top-1 指标则显示 dual-softmax 在本验证集上低于 forward 基线。

## 几何深度诊断

`metrics.json` 还保留由匹配卫星位置和相机位姿回投得到的 camera-z 对比 LiDAR 的 coarse/refine 几何诊断。该组数值不做 confidence scale 校正，与上面的主深度估计结果分开记录。

## 复现命令

```bash
conda activate bevsplat
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/evaluate_matching_ab.py \
  --config configs/config_confidence.yaml \
  --checkpoint outputs_0923/stage1_refine/checkpoints/stage1_latest.pth \
  --confidence_checkpoint outputs_1021/stage1_confidence/checkpoints/confidence_best.pth \
  --manifest outputs_0923/stage1_refine/kitti360_val400_manifest.json \
  --output_dir outputs_1026/b1_dual_softmax \
  --geometry_threshold_px 10 \
  --device cpu --batch_size 8 --num_workers 0
```
