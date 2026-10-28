# Stage-1 Front-to-Satellite Correspondence

## Code layout

```text
models/       backbone, matcher, refiner, confidence and Stage-1 model
data/         dataset adapter and geometry/target generation
training/     training entrypoints and training losses
evaluation/   shared depth metrics and anchor-scale estimators
utils/        config loading and checkpoint helpers
scripts/      evaluation, visualization, data-preparation and utility CLIs
```

Commands below are run from `/mnt/mydisk/fkx/DC`; CLI modules live under
`models.stage1.scripts` or `models.stage1.training`.

## Train

From `/mnt/mydisk/fkx/DC`:

```bash

第一阶段 coarse训练：

smoke test
python -m models.stage1.training.train \
  --config models/stage1/configs/config_smoke.yaml \
  --stage coarse \
  --output_dir models/stage1/outputs/stage1_coarse_smoke \
  --max_steps 100 \
  --overfit_batches 5

DINO test
python -m models.stage1.training.train \
  --config models/stage1/configs/config.yaml \
  --output_dir models/stage1/outputs/stage1_coarse_dino_sanity \
  --max_steps 100 \
  --overfit_batches 5 \
  --max_val_batches 20

正式训练
python -m models.stage1.training.train \
  --config models/stage1/configs/config.yaml \
  --output_dir models/stage1/outputs/stage1_coarse


第二阶段 refiner 训练：

python -m models.stage1.training.train \
  --config models/stage1/configs/config_refine.yaml \
  --init_checkpoint models/stage1/outputs/stage1_coarse/checkpoints/stage1_latest.pth


说明：
- --init_checkpoint：从 coarse 权重初始化一个新的 refine 训练，不继承 epoch/optimizer。
- --checkpoint：恢复同一个训练任务的 latest 断点。
```

第三阶段 confidence 训练：

confidence 阶段只加载并冻结已有 coarse+refine 网络，训练新的 scale-weighting
confidence head。它是接收前视/卫星特征的独立 MLP，不会微调 Stage-1 的 refiner。
训练/验证 manifest 必须同时提供单目深度
(`unidepth_depth_path`、`depth_path` 或 `da3_depth_path`) 和 LiDAR 深度
(`lidar_depth_path`)。

```bash
python -m models.stage1.training.train_confidence \
  --config models/stage1/configs/config_confidence.yaml \
  --base_checkpoint models/stage1/outputs_0923/stage1_refine/checkpoints/stage1_latest.pth \
  --manifest models/stage1_anchor/outputs/260703/kitti360_train_0000_manifest.json \
  --val_manifest models/stage1/outputs_0923/stage1_refine/kitti360_val400_manifest.json \
  --output_dir models/stage1/outputs_0923/stage1_confidence
```

今后稠密单目深度的主指标统一采用 confidence head 加权的几何 scale，公式为
`scale = exp(sum_i normalized_confidence_i * log(anchor_z_i / mono_anchor_z_i))`。
评估入口会在同一批固定 LiDAR 有效像素上报告该主指标；等权、算术加权和 Stage-1
certainty Top-20 只作为对照消融。匹配位置回投得到的 camera-z 指标单独记为几何诊断。

训练产生的实际权重文件名带有 `_epoch_NNN_step_NNNNNN`；`stage1_latest.pth`、
`confidence_latest.pth` 等稳定名称保留为指向当前版本化权重的符号链接，便于旧命令继续加载。
新的 confidence checkpoint 还会保存配套 Stage-1 checkpoint 的 SHA-256，续训和评估时会检查配对。

评估时会在同一批 LiDAR 有效像素上比较等权 scale 与 confidence 加权 scale：

```bash
python -m models.stage1.scripts.evaluate_confidence \
  --config models/stage1/configs/config_confidence.yaml \
  --base_checkpoint models/stage1/outputs_0923/stage1_refine/checkpoints/stage1_latest.pth \
  --confidence_checkpoint models/stage1/outputs_1021/stage1_confidence/checkpoints/confidence_best.pth \
  --manifest models/stage1/outputs_0923/stage1_refine/kitti360_val400_manifest.json \
  --output_dir models/stage1/outputs_0923/stage1_confidence_eval
```

`confidence_weighted` 是主深度结果。confidence 只用于全图 scale 估计的相对权重，不解释为匹配正确概率；候选锚点不使用 matchability/certainty 门控，也不做 Top-K、RANSAC 或二次尺度对齐。

For a fast shape/debug run, set `model.backbone_type: conv` in a temporary config and run:

```bash
python -m models.stage1.training.train --config models/stage1/configs/config.yaml --output_dir models/stage1/outputs/stage1_smoke --max_steps 2 --overfit_batches 1
```

## Evaluate

中文测评协议、正式结果、消融实验和坐标变换约定见
[`docs/测评汇总.md`](docs/测评汇总.md)。

稠密单目深度主指标通过 `scripts/evaluate_confidence.py` 计算，统一采用 learned confidence-head
加权的几何 scale 校正；`scripts/evaluate.py`、`scripts/evaluate_depth.py` 和 B1 的匹配位置深度结果是
匹配/几何诊断，不代替该主指标。

```bash
python -m models.stage1.scripts.evaluate \
  --config models/stage1/configs/config.yaml \
  --manifest /mnt/mydisk/data/public_data/DC/KITTI-360/stage1_test_manifest.json \
  --checkpoint models/stage1/outputs/stage1_coarse/checkpoints/stage1_latest.pth \
  --output_dir models/stage1/outputs/stage1_test_eval 
```

Outputs include coarse accuracy, coarse GT distance, refined pixel/metric error, separate matchability and certainty statistics, and visualizations.

TensorBoard uses `matchability/*` for the coarse matchability head and
`certainty/*` for the final refine head. The corresponding binary losses are
`loss/matchability` and `loss/certainty`.

When continuing a run whose `metrics.jsonl` uses the legacy field names,
training writes the new schema to `metrics_v2.jsonl` instead of mixing both
schemas in one file.
The same compatibility rule uses `tensorboard_v2/` for the corresponding new
event stream.

## Visualize

GT geometry sanity check:

```bash
python -m models.stage1.scripts.visualize --config models/stage1/configs/config.yaml --mode gt --output_dir models/stage1/outputs/vis
```

Prediction visualization:

```bash
python -m models.stage1.scripts.visualize \
  --config models/stage1/configs/config.yaml \
  --checkpoint outputs/stage1/checkpoints/stage1_latest.pth \
  --mode pred \
  --output_dir models/stage1/outputs/vis
```

Confidence weight heatmap visualization:

```bash
python -m models.stage1.scripts.visualize \
  --config models/stage1/configs/config_confidence.yaml \
  --base_checkpoint models/stage1/outputs_0923/stage1_refine/checkpoints/stage1_latest.pth \
  --confidence_checkpoint models/stage1/outputs_1021/stage1_confidence/checkpoints/confidence_best.pth \
  --manifest models/stage1/outputs_0923/stage1_refine/kitti360_val400_manifest.json \
  --sample_id 0000010926 \
  --mode confidence \
  --device cpu \
  --output_dir models/stage1/outputs_1021/visualizations/confidence_0000010926
```

生成 `confidence_weights_interactive.html`。热力图颜色表示每张图内归一化后的相对权重
`w_i / max(w)`；同一锚点在前视图和预测卫星匹配点上使用同一颜色，并在合并视图中用连线对应。
HTML 中的 `c_i`、归一化 `w_i`、相机 z-depth、单目 z-depth 和 z 比值可点击查看。

此外还支持：
- --manifest：指定 manifest 文件
- --sample_id：指定样本 ID
- --sample_index：按 manifest 下标指定样本
```bash

python -m models.stage1.scripts.visualize \
  --config models/stage1/configs/config.yaml \
  --manifest /mnt/mydisk/data/public_data/DC/KITTI-360/stage1_test_manifest.json \
  --sample_id 0000003555 \
  --checkpoint models/stage1/outputs/stage1_coarse/checkpoints/stage1_latest.pth \
  --mode pred \
  --device cpu \
  --output_dir models/stage1/outputs/stage1_coarse/interactive_test_3555
```

## 续训命令

```bash
  python -m models.stage1.training.train \
    --config models/stage1/configs/config.yaml \
    --output_dir /tmp/my_stage1_run \
    --checkpoint /tmp/my_stage1_run/checkpoints/stage1_latest.pth
```

对已有 checkpoint 重新评估即可生成完整诊断：

python -m models.stage1.scripts.evaluate \
  --config models/stage1/configs/config.yaml \
  --checkpoint models/stage1/outputs/stage1_coarse/checkpoints/stage1_best.pth \
  --output_dir models/stage1/outputs/stage1_coarse/diagnostics_best_loss

再对 stage1_latest.pth 或其他 checkpoint 使用不同的 --output_dir，比较上述指标即可确认概率校准是否变差。

## 候选对应图空间一致性

粗匹配的候选图在前视点与卫星点的相似度 logits 之后、粗坐标 argmax 之前运行。每个粗前视点先保留 `matcher.candidate_graph.candidate_count` 个经空间 NMS 分散的卫星峰值；局部网格邻居和固定跨区域邻居之间建立稀疏候选边。图输出的残差会写回原始 logits，再重新计算前向概率、可选 dual-softmax 分数、matchability 和粗坐标。`enabled: false` 时沿用原 logits 与解码流程。

三个可直接运行的消融配置：

- `configs/config_graph_original.yaml`：关闭候选图，作为原结构。
- `configs/config_graph_no_geometry.yaml`：开启候选图，只移除几何边特征。
- `configs/config_graph_full.yaml`：候选图加单目深度几何边特征。

深度字段通过 `unidepth_depth_path`、`front_depth_path`、`depth_path` 或 `da3_depth_path` 读取。`matcher.candidate_graph.depth_type` 必须显式设为 `camera_z`、`ray_distance`、`inverse_camera_z` 或 `inverse_ray_distance`，`depth_units` 显式标记 `metric` 或 `arbitrary`。坐标约定沿用 OpenCV 相机轴：x 向右、y 向下、z 向前；射线距离会除以射线范数转成 z 深度，逆深度先取倒数。每个像素深度沿相机射线形成三维点，再经固定相机外参转换并提取局部水平坐标；不把可见点强制投到地平面。默认 `depth_scale_mode: robust` 使用当前 top-1 预测和稀疏查询边，对每张图估计一个共享正尺度；仅当单目深度已校准成米制时才设 `depth_units: metric` 与 `depth_scale_mode: metric`。若深度只有排序意义，将 `depth_relation` 设为 `ranking_only`，几何边特征会关闭，保留视觉图消息。

卫星图像坐标为 `(u 向右, v 向下)`。仅在配置显式声明 `satellite_orientation_default: north_up` 时，缺少逐图方向轴的卫星图按 `x = center_x + (u-center_u)*meters_per_pixel`、`y = center_y - (v-center_v)*meters_per_pixel` 转为世界 XY。自车朝向图须同时给出单位且近似正交的 `satellite_forward_world_xy` 和 `satellite_right_world_xy`，两者定义 `(v 向上)` 和 `(u 向右)` 的世界方向。前视点只使用固定标定外参 `camera_to_vehicle` 转入车体坐标，不读取 `vehicle_to_world`。KITTI 配置显式设 `vehicle_horizontal_axes: [1, 0]`，即先取车体右向、再取前向；候选 top-1 预测上用鲁棒估计得到一组共享平面旋转和正尺度（metric 模式固定尺度为 1），不使用真值位姿，也不按边单独拟合。缺少地图中心/分辨率或明确地图方向时，图边几何有效掩码关闭；不会把默认分辨率当作已验证输入。单目深度允许添加 `depth_valid_mask_path` 与 `depth_confidence_path`，缺失深度只屏蔽几何边，不会屏蔽视觉候选匹配。

训练在原粗对应 CE 外增加候选覆盖监督和仅对有效、已覆盖 LiDAR 标签计算的鲁棒结构损失；日志包含 `candidate_coverage`。refine/joint 阶段可用 `lambda_refine_structure` 对细化结果加世界平面相对布局约束。无效/未知标签以及正确栅格不在候选集内的查询不会进入候选结构监督。老 checkpoint 可以载入，只有 `matcher.candidate_graph.*` 新键允许缺失；其他键不匹配会报错。

从仓库父目录 `/mnt/mydisk/fkx/DC` 开始训练与评估：

```bash
python -m models.stage1.training.train \
  --config models/stage1/configs/config_graph_full.yaml \
  --output_dir models/stage1/outputs/stage1_graph_full

python -m models.stage1.scripts.evaluate \
  --config models/stage1/configs/config_graph_full.yaml \
  --checkpoint models/stage1/outputs/stage1_graph_full/checkpoints/stage1_latest.pth \
  --manifest /mnt/mydisk/data/public_data/DC/KITTI-360/stage1_val_manifest.json \
  --output_dir models/stage1/outputs/stage1_graph_full_eval
```
