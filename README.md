# Morphology-Conditioned Feasibility Prior (MCFP)

本仓库实现了一个面向串联机械臂的条件可达性距离场（conditional reachability distance field）训练与推理流程。当前工程由两个核心场组成：

- `Position-SDF`：定义在工作空间位置上的带符号距离场。
- `Orientation-SDF`：定义在末端姿态上的带符号测地角距离场，并以固定锚点位置为条件。

两者组合后，支持以下三类任务：

- 位置可达性查询与位置回拉投影；
- 固定锚点处的姿态可达性查询与姿态回拉投影；
- 先修正位置、再修正姿态的联合位姿修复。

当前仓库中的配置、路径示例和默认数据组织主要围绕 `aubo_i5_3mm` 这一套 Aubo i5 数据展开。

原始数据生成流程已经迁移到独立的 ROS2 工程中。本仓库当前主要负责：

- 将导出的原始 HDF5 数据整理为训练可用的数据集；
- 从 URDF/SRDF 提取机械臂构型与碰撞几何信息；
- 训练 Position-SDF 与 Orientation-SDF 模型；
- 执行位置、姿态和联合位姿推理；
- 提供训练验证、数据统计与可视化工具。

## 仓库结构

- `configs/`：数据准备、训练和推理所用的 YAML 配置。
- `mcfp/`：核心 Python 包，包含数据加载、模型和训练逻辑。
- `scripts/`：命令行入口脚本与可视化工具。
- `runs/`：训练产生的 checkpoint、metrics 和中间输出。
- `outputs/`：验证结果和可视化导出结果。
- `data/`、`datasets/`、`docs/`：本地开发资产、原始数据和设计文档。开发时通常会依赖这些目录，但开源分支不一定需要跟踪它们。

## 环境准备

以下命令默认都在仓库根目录执行。

### 本地环境

```bash
conda activate mcfp
cd /home/solegua/mcfp
```

### 服务器环境

```bash
cd /home/ghc/MCFP
conda activate /home/ghc/.conda/envs/mcfp
tmux new -s mcfp
```

从 `tmux` 会话分离：

```bash
Ctrl+b d
```

重新连接：

```bash
tmux attach -t mcfp
```

## 配置文件说明

- `configs/prepare_data.yaml`：原始输入路径、处理后输出路径、数据切分和 morphology 提取参数。
- `configs/train_pos.yaml`：Position-SDF 训练配置。
- `configs/train_orient.yaml`：Orientation-SDF 训练配置。
- `configs/infer_pos.yaml`：Position-SDF 推理和位置场可视化配置。
- `configs/infer_orient.yaml`：Orientation-SDF 推理配置。
- `configs/infer_pose.yaml`：联合位姿推理配置。

在训练或推理前，建议先检查这些配置中的：

- 数据集路径是否正确；
- checkpoint 路径是否存在；
- `run_dir` 是否与当前实验目录一致。

## 典型流程

1. 准备训练数据和 morphology 元数据。
2. 训练 Position-SDF 模型。
3. 训练 Orientation-SDF 模型。
4. 运行位置、姿态或联合位姿推理。
5. 使用验证和可视化脚本做 sanity check。

## 命令说明

### 1. 数据准备

将导出的原始 HDF5 数据和机器人描述文件整理为训练可用的数据格式。

```bash
python -m scripts.prepare_data --config configs/prepare_data.yaml
```

这一步通常会生成以下核心文件：

- `data/<robot>/dataset_pos.h5`
- `data/<robot>/orientation.h5`
- `data/<robot>/meta/morphology_spec.h5`

### 2. 模型训练

训练 Position-SDF 模型：

```bash
python -m scripts.train_pos --config configs/train_pos.yaml
```

作用：基于处理后的体素级位置数据训练位置可达性距离场。

训练 Orientation-SDF 模型：

```bash
python -m scripts.train_orient --config configs/train_orient.yaml
```

作用：基于固定锚点的姿态数据训练条件姿态距离场。

### 3. 推理

运行位置可达性推理，并可选择显示投影路径：

```bash
python -m scripts.infer_pos --config configs/infer_pos.yaml --vis
```

作用：判断一个查询位置是否可达；如果不可达，则沿梯度方向做迭代回拉。

命令行直接覆盖查询位置的例子：

```bash
python -m scripts.infer_pos --config configs/infer_pos.yaml --pos 0.2 0.2 0.7 --vis
```

运行固定锚点处的姿态可达性推理：

```bash
python -m scripts.infer_orient --config configs/infer_orient.yaml
```

作用：判断某个锚点位置下给定姿态是否可达，并在需要时执行姿态回拉。

运行联合位姿推理：

```bash
python -m scripts.infer_pose --config configs/infer_pose.yaml --vis
```

作用：先修正位置可达性，再在修正后的锚点上继续修正姿态。

命令行直接覆盖查询位姿的例子：

```bash
python -m scripts.infer_pose --config configs/infer_pose.yaml --pose 0.2 0.2 0.7 180 180 270 --vis
```

### 4. 数据检查与可视化

统计处理后位置数据集的整体分布和 bucket 结构：

```bash
python -m scripts.stats_dataset_pos --path data/aubo_i5_3mm/dataset_pos.h5
```

作用：查看边界体素、非边界体素、距离分桶和索引结构的整体统计。

抽样一个 Position-SDF 训练 batch 并检查采样器分布：

```bash
python -m scripts.sample_batch_pos_stats --config configs/train_pos.yaml --split 1
```

作用：检查训练 batch 的 boundary / non-boundary 配比、正负样本平衡和 tier 分布。

可视化处理后的位置数据集：

```bash
python -m scripts.visualize_dataset_pos_pv --path data/aubo_i5_3mm/dataset_pos.h5 --show-slices
```

作用：用 PyVista 查看边界体素、网格过滤结果和 bucket 分布。

可视化学习到的 Position-SDF 场：

```bash
python -m scripts.visualize_pos_field_mpl --config configs/infer_pos.yaml
```

作用：在给定立方体采样范围内评估训练后的 Position-SDF 网络输出，并显示结果。

带切面显示的示例：

```bash
python -m scripts.visualize_pos_field_mpl \
  --config configs/infer_pos.yaml \
  --show-slices \
  --slice-thickness 0.02 \
  --slice-center 0 0 0
```

可视化某个锚点的姿态数据：

```bash
python -m scripts.visualize_dataset_orient_anchor
```

作用：输出单个锚点的姿态覆盖、twist 覆盖和若干 sanity-check 视图。这个脚本使用文件顶部 `CONFIG` 中的参数。

### 5. 验证与训练监控

对训练结果进行离线验证，并输出 JSON 摘要：

```bash
python -m scripts.validate_training --only both --split val --out-json outputs/val_metrics.json
```

作用：在训练/验证/测试划分上快速评估 Position-SDF 与 Orientation-SDF 的预测表现。

监控训练产生的 `metrics.csv`，持续刷新 loss 图：

Position-SDF 训练监控示例：

```bash
python -m scripts.watch_loss \
  --metrics runs/pos_sdf/exp001/metrics.csv \
  --out runs/pos_sdf/exp001/loss.png \
  --interval 5
```

Orientation-SDF 训练监控示例：

```bash
python -m scripts.watch_loss \
  --metrics runs/orient_sdf/exp001/metrics.csv \
  --out runs/orient_sdf/exp001/loss.png \
  --interval 5
```

请根据你的实际实验目录调整上面的 `exp001` 路径。

## 服务器训练示例

在服务器上启动一个训练会话：

```bash
cd /home/ghc/MCFP
conda activate /home/ghc/.conda/envs/mcfp
tmux new -s mcfp
```

在会话中启动训练：

```bash
python -m scripts.train_pos --config configs/train_pos.yaml
python -m scripts.train_orient --config configs/train_orient.yaml
```

如果你希望单独开一个终端监控 loss，可以在另一个 shell 中执行上面的 `scripts.watch_loss` 命令。
