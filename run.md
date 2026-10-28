# Stage-1 Front-to-Satellite Correspondence

## 多卡分布式训练

主匹配模型使用 4 张 GPU 时，从 `/mnt/mydisk/fkx/DC` 启动：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m models.stage1.training.train \
  --config models/stage1/configs/config_graph_full.yaml \
  --output_dir models/stage1/outputs/stage1_ddp
```

`config_graph_full.yaml` 中的 `training.batch_size` 是**每张 GPU 的 batch size**，所以全局 batch size 等于该值乘以 GPU 数量。验证、TensorBoard、metrics、可视化和 checkpoint 只由 rank 0 写入。启动续训时所有 rank 使用同一个 `--checkpoint` 路径。

confidence head 训练也支持相同的 DDP 启动方式：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m models.stage1.training.train_confidence \
  --config models/stage1/configs/config_confidence.yaml \
  --base_checkpoint models/stage1/outputs/stage1_refine/checkpoints/stage1_latest.pth \
  --output_dir models/stage1/outputs/stage1_confidence_ddp
```

## Train
From `/mnt/mydisk/fkx/DC`:

0917第一阶段 coarse训练：

python -m models.stage1.training.train \
  --config models/stage1/configs/config.yaml \
  --output_dir models/stage1/outputs/stage1_coarse

总共71epoch

0921 第二阶段 refiner 训练：

python -m models.stage1.training.train \
  --config models/stage1/configs/config_refine.yaml \
  --init_checkpoint models/stage1/outputs/stage1_coarse/checkpoints/stage1_latest.pth



1020
隐式训练confidence(之前的命名可能没统一，现已更改)
训练方式：
  - coarse/joint 训练：logit_scale 可训练
  - refine 训练：matcher 已冻结，因此固定
  - confidence 训练：整个 coarse+refine 网络冻结，因此固定


cd /mnt/mydisk/fkx/DC

训练命令
python -m models.stage1.training.train_confidence \
--config models/stage1/configs/config_confidence.yaml \
--base_checkpoint models/stage1/outputs_0923/stage1_refine/checkpoints/stage1_latest.pth \
--manifest /mnt/mydisk/data/public_data/DC/KITTI-360/stage1_train_manifest.json \
--val_manifest models/stage1/outputs_0923/stage1_refine/kitti360_val400_manifest.json \
--output_dir models/stage1/outputs_1021/stage1_confidence

续训命令
python -m models.stage1.training.train_confidence \
--config models/stage1/configs/config_confidence.yaml \
--base_checkpoint models/stage1/outputs_0923/stage1_refine/checkpoints/stage1_latest.pth \
--checkpoint models/stage1/outputs_1020/stage1_confidence/checkpoints/confidence_latest.pth \
--manifest /mnt/mydisk/data/public_data/DC/KITTI-360/stage1_train_manifest.json \
--val_manifest models/stage1/outputs_0923/stage1_refine/kitti360_val400_manifest.json \
--output_dir models/stage1/outputs_1020/stage1_confidence

验证命令
python -m models.stage1.scripts.evaluate_confidence \
--config models/stage1/configs/config_confidence.yaml \
--base_checkpoint models/stage1/outputs_0923/stage1_refine/checkpoints/stage1_latest.pth \
--confidence_checkpoint models/stage1/outputs_1021/stage1_confidence/checkpoints/confidence_best.pth \
--manifest models/stage1/outputs_0923/stage1_refine/kitti360_val400_manifest.json \
--output_dir models/stage1/outputs_1026/depth_metrics_confidence_weighted

该评估输出中的 `confidence_weighted` 是稠密单目深度主指标，使用 learned confidence head
对锚点 scale ratio 做 log-space 加权几何平均。等权、算术 scale 和 certainty Top-20 是消融；
匹配位置回投得到的 camera-z 则是单独的几何诊断。

confidence 权重可视化（前视锚点、预测卫星匹配点及对应连线使用同一热力颜色）：

```bash
cd /mnt/mydisk/fkx/DC
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

打开 `models/stage1/outputs_1021/visualizations/confidence_0000010926/confidence_weights_interactive.html`，可点击锚点查看具体 `c_i`、归一化权重和深度信息。


