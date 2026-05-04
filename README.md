# Reachability Dataset Workspace

这个 ROS 2 工作空间用于实现 `RDF_Complete_Memo_Obsidian_v12.md` 第二章中的数据采样与标注流程。当前主线目标是：

- 为机械臂生成位置可达性数据集
- 在位置数据集基础上继续生成姿态可达性数据集
- 将通用生成逻辑与机器人专属资产分离，便于后续扩展到其它机械臂

当前默认配置面向 `Aubo i5`，但目录结构已经按“通用能力 + 机器人专属配置”整理完成。

## 当前工程的作用

这个工作空间不是做网络训练的，它主要负责离线数据生产。核心流程包括：

1. 读取机器人 URDF / SRDF / 碰撞模型
2. 基于 MoveIt Core 做 FK、IK 和自碰撞检查
3. 采样工作空间体素并生成位置数据集
4. 在位置数据集基础上生成 anchor-based 姿态数据集
5. 输出为后续网络训练可直接消费的 HDF5 或辅助数组文件

当前主生成器不依赖 `ros2 launch` 先启动一个机器人服务节点，而是通过 CLI 在进程内部直接加载模型文件并完成计算。

## 目录结构

- `src/core/reachability_cli`
  - 通用离线数据生成核心
  - 包含 `dataset_generator_cli`、`orientation_dataset_cli`、FK/IK 调试工具等
- `src/robots/aubo/aubo_description`
  - Aubo 的 URDF / Xacro / mesh / RViz 资源
- `src/robots/aubo/aubo_moveit_config`
  - Aubo 的 SRDF、关节限制、MoveIt 配置
- `src/robots/aubo/aubo_reachability`
  - 历史 ROS 2 service 包，当前主离线生成链路不依赖它
- `tools/data_gen/configs/robots/<vendor>/<model>/`
  - 数据生成配置
- `tools/data_gen/scripts/runners/`
  - 本地和服务器的一键入口脚本
- `tools/data_gen/scripts/visualize/`
  - 数据集可视化脚本
- `tools/data_gen/scripts/postprocess/`
  - 后处理脚本
- `tools/data_gen/outputs/`
  - 生成数据输出目录，默认不上传 Git
- `tools/model_validate/`
  - 预留给后续网络验证功能的骨架目录

## 当前默认配置

当前 Aubo i5 的核心配置文件：

- 位置数据：`tools/data_gen/configs/robots/aubo/aubo_i5/position_3mm.yaml`
- 姿态数据（V1.3 正式 1024 anchors）：`tools/data_gen/configs/robots/aubo/aubo_i5/orientation_3mm.yaml`

当前默认输出目录：

- `tools/data_gen/outputs/aubo/aubo_i5/voxel_3mm/`

主要输出文件通常包括：

- `dataset.h5`
- `dataset_orient.h5`

## Aubo i5 正式数据生成启动方式

统一使用一个脚本启动数据挖掘。默认机器人是 Aubo i5：

```bash
cd /home/ninesoo/ros2_workspace
bash tools/data_gen/scripts/runners/run_pipeline_local.sh
```

默认不带参数等价于 `all`，会自动编译数据生成相关包，然后先跑 3mm 位置 SDF，再基于生成的 `dataset.h5` 继续跑 V1.3 1024-anchor 姿态 SDF。

可选模式：

```bash
# 完整流程：位置 -> 姿态，默认模式
bash tools/data_gen/scripts/runners/run_pipeline_local.sh all

# 只跑位置 SDF，生成 dataset.h5
bash tools/data_gen/scripts/runners/run_pipeline_local.sh pos

# 只跑姿态 SDF，要求位置阶段的 dataset.h5 已存在
bash tools/data_gen/scripts/runners/run_pipeline_local.sh orient

# 只跑可视化脚本
bash tools/data_gen/scripts/runners/run_pipeline_local.sh vis
```

脚本当前固定使用 Aubo 正式配置：

- 位置配置：`tools/data_gen/configs/robots/aubo/aubo_i5/position_3mm.yaml`
- 姿态配置：`tools/data_gen/configs/robots/aubo/aubo_i5/orientation_3mm.yaml`
- 输出目录：`tools/data_gen/outputs/aubo/aubo_i5/voxel_3mm/`
- 运行日志：`tools/data_gen/outputs/aubo/aubo_i5/voxel_3mm/logs/`

脚本默认导出 32 线程 OpenMP 设置，并通过 `/usr/bin/time -v` 记录资源使用。需要临时改线程数时，可以在命令前覆盖环境变量：

```bash
OMP_NUM_THREADS=24 bash tools/data_gen/scripts/runners/run_pipeline_local.sh all
```

如果已经编译过，只想直接跑，可以跳过脚本内编译：

```bash
BUILD_BEFORE_RUN=false bash tools/data_gen/scripts/runners/run_pipeline_local.sh orient
```

同一个脚本也可以通过 `ROBOT` 切换到其它机械臂：

```bash
# RealMan RM65 完整流程
ROBOT=realman bash tools/data_gen/scripts/runners/run_pipeline_local.sh all

# Franka Panda 完整流程
ROBOT=franka bash tools/data_gen/scripts/runners/run_pipeline_local.sh all

# 只跑 Franka 姿态阶段
ROBOT=franka bash tools/data_gen/scripts/runners/run_pipeline_local.sh orient
```

## 本地使用方法

本地一键入口脚本：

- `tools/data_gen/scripts/runners/run_pipeline_local.sh`

支持四种模式：

- `all`：位置数据 + 姿态数据一起跑
- `pos`：只跑位置数据
- `orient`：只跑姿态数据
- `vis`：只跑可视化

常用命令：

```bash
bash tools/data_gen/scripts/runners/run_pipeline_local.sh
bash tools/data_gen/scripts/runners/run_pipeline_local.sh pos
bash tools/data_gen/scripts/runners/run_pipeline_local.sh orient
```

脚本会自动做这些事情：

1. 进入工作空间
2. 尝试退出当前 Conda 环境
3. `source /opt/ros/humble/setup.bash`
4. 自动清理失效的 `reachability_cli` CMake 缓存
5. 执行 `colcon build --packages-select aubo_description aubo_moveit_config reachability_cli --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release`
6. `source install/setup.bash`
7. 调用对应的 CLI 生成器

如果你只想手动执行核心命令，也可以这样跑：

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select aubo_description aubo_moveit_config reachability_cli --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

ros2 run reachability_cli dataset_generator_cli \
  --config tools/data_gen/configs/robots/aubo/aubo_i5/position_3mm.yaml

ros2 run reachability_cli orientation_dataset_cli \
  --config tools/data_gen/configs/robots/aubo/aubo_i5/orientation_3mm.yaml
```

注意：

- `orientation_3mm.yaml` 依赖位置阶段先生成的 `dataset.h5`
- 所以通常先跑 `pos`，再跑 `orient`

## 服务器使用方法

服务器入口脚本：

- `tools/data_gen/scripts/runners/run_pipeline_zyun.sh`

它默认假设：

- 你进入脚本之前，当前 shell 已经准备好了服务器上的 ROS / 虚拟环境
- 脚本不会强制改动你现有的服务器环境

常用命令：

```bash
bash tools/data_gen/scripts/runners/run_pipeline_zyun.sh pos
bash tools/data_gen/scripts/runners/run_pipeline_zyun.sh orient
bash tools/data_gen/scripts/runners/run_pipeline_zyun.sh all
```

这个脚本默认工作目录是：

```bash
/home/user/Zyun/ros2_workspace
```

它会做这些事情：

1. 进入服务器工作空间
2. 可选清理 `build/ install/ log/`
3. 自动检测并清理旧的 `reachability_cli` 构建缓存
4. 执行 `colcon build`
5. `source install/setup.bash`
6. 调用位置或姿态数据生成器

如果你想沿用你以前在服务器上手动跑的方式，等价命令仍然可以写成：

```bash
rm -rf build/ install/ log/
colcon build --symlink-install
source /home/user/Zyun/ros2_workspace/install/setup.bash

ros2 run reachability_cli dataset_generator_cli \
  --config tools/data_gen/configs/robots/aubo/aubo_i5/position_3mm.yaml

ros2 run reachability_cli orientation_dataset_cli \
  --config tools/data_gen/configs/robots/aubo/aubo_i5/orientation_3mm.yaml
```

## RealMan RM65 正式数据生成命令

RealMan RM65 使用与 Aubo 当前正式流程一致的 3mm 位置网格和 V1.3 1024-anchor 姿态挖掘配置。

核心配置文件：

- 位置数据：`tools/data_gen/configs/robots/realman/rm65/position_3mm.yaml`
- 姿态数据（V1.3 正式 1024 anchors）：`tools/data_gen/configs/robots/realman/rm65/orientation_3mm.yaml`

默认输出目录：

- `tools/data_gen/outputs/realman/rm65/voxel_3mm/`

### 本机运行

```bash
cd /home/ninesoo/ros2_workspace
source /opt/ros/humble/setup.bash
colcon build \
  --packages-select rm_description rm_moveit_config reachability_cli \
  --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

export OMP_NUM_THREADS=32
export OMP_DYNAMIC=false
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p tools/data_gen/outputs/realman/rm65/voxel_3mm/logs

taskset -c 0-31 /usr/bin/time -v ros2 run reachability_cli dataset_generator_cli \
  --config tools/data_gen/configs/robots/realman/rm65/position_3mm.yaml \
  2>&1 | tee tools/data_gen/outputs/realman/rm65/voxel_3mm/logs/rm65_position_3mm_$(date +%Y%m%d_%H%M%S).log

taskset -c 0-31 /usr/bin/time -v ros2 run reachability_cli orientation_dataset_cli \
  --config tools/data_gen/configs/robots/realman/rm65/orientation_3mm.yaml \
  2>&1 | tee tools/data_gen/outputs/realman/rm65/voxel_3mm/logs/rm65_orientation_3mm_1024_$(date +%Y%m%d_%H%M%S).log
```

### 192.168.39.180 运行

部署目录固定为：

```bash
/home/user/data/ros2_workspace
```

在 180 机器的本地终端中使用 `tmux` 运行，避免长时间跑数时终端关闭导致任务中断。

推荐直接用统一 runner 启动完整流程：

```bash
tmux new -s realman_data 'bash -lc '"'"'
cd /home/user/data/ros2_workspace
WORKSPACE=/home/user/data/ros2_workspace ROBOT=realman \
  bash tools/data_gen/scripts/runners/run_pipeline_local.sh all
exec bash
'"'"''
```

只跑位置或只跑姿态：

```bash
WORKSPACE=/home/user/data/ros2_workspace ROBOT=realman bash tools/data_gen/scripts/runners/run_pipeline_local.sh pos
WORKSPACE=/home/user/data/ros2_workspace ROBOT=realman bash tools/data_gen/scripts/runners/run_pipeline_local.sh orient
```

`tmux` 常用操作：

- 暂时离开但不停止任务：按 `Ctrl-b`，松开后按 `d`
- 重新进入会话：`tmux attach -t realman_data`
- 查看现有会话：`tmux ls`
- 跑完后退出会话：在 `tmux` 中执行 `exit`
- 确认要强制关闭该会话时：`tmux kill-session -t realman_data`

## Franka Emika Panda 正式数据生成命令

Franka Emika Panda 使用同一套 3mm 位置网格和 V1.3 1024-anchor 姿态挖掘流程。

Franka 在 180 机器上只有约 62GiB 内存，位置阶段使用更紧凑的 per-voxel FK seed 配置：
`sampling.voxel_reservoir_max: 2`。这仍会保留每个命中体素的少量 IK warm-start seed，同时避免 8 seed 配置在 64GB 机器上被 OOM kill。

核心配置文件：

- 位置数据：`tools/data_gen/configs/robots/franka_emika_panda/panda/position_3mm.yaml`
- 姿态数据（V1.3 正式 1024 anchors）：`tools/data_gen/configs/robots/franka_emika_panda/panda/orientation_3mm.yaml`

默认输出目录：

- `tools/data_gen/outputs/franka_emika_panda/panda/voxel_3mm/`

### 192.168.39.180 运行

如果 `realman_data` 还在跑，不要启动 Franka 正式生成任务，避免两个 32 线程任务抢 CPU。先查看或回到 RealMan 会话：

```bash
tmux ls
tmux attach -t realman_data
```

等 RealMan 跑完后，可以在 180 机器的本地终端直接执行下面这一条命令。它会新建 `franka_data` tmux 会话，自动完成环境初始化，先跑位置 SDF；位置阶段成功生成 `dataset.h5` 后，再继续跑 1024-anchor 姿态 SDF。

```bash
tmux new -s franka_data 'bash -lc '"'"'
cd /home/user/data/ros2_workspace
WORKSPACE=/home/user/data/ros2_workspace ROBOT=franka \
  bash tools/data_gen/scripts/runners/run_pipeline_local.sh all
exec bash
'"'"''
```

只跑位置或只跑姿态：

```bash
WORKSPACE=/home/user/data/ros2_workspace ROBOT=franka bash tools/data_gen/scripts/runners/run_pipeline_local.sh pos
WORKSPACE=/home/user/data/ros2_workspace ROBOT=franka bash tools/data_gen/scripts/runners/run_pipeline_local.sh orient
```

`tmux` 常用操作：

- 暂时离开但不停止任务：按 `Ctrl-b`，松开后按 `d`
- 重新进入会话：`tmux attach -t franka_data`
- 查看现有会话：`tmux ls`
- 跑完后退出会话：在 `tmux` 中执行 `exit`
- 确认要强制关闭该会话时：`tmux kill-session -t franka_data`

## 可视化与后处理

常用脚本包括：

- `tools/data_gen/scripts/visualize/plot_occupancy_voxels.py`
- `tools/data_gen/scripts/visualize/plot_orient_samples.py`
- `tools/data_gen/scripts/visualize/plot_anchor_distribution.py`
- `tools/data_gen/scripts/postprocess/pack_npz.py`
- `tools/data_gen/scripts/postprocess/fix_orient_method3_phi.py`

这些脚本主要用于检查生成结果、观察 anchor 分布、导出 legacy 格式或修补历史数据。

## 后续加新机械臂

建议按下面的方式扩展：

1. 在 `src/robots/<vendor>/` 下加入该机械臂的 description 包和 moveit_config 包
2. 在 `tools/data_gen/configs/robots/<vendor>/<model>/` 下新增位置与姿态配置
3. 在配置中修改：
   - `robot.urdf`
   - `robot.srdf`
   - `group_name`
   - `base_link`
   - `ee_link`
   - `joint_names`
   - `output.*`
4. 使用相同的 runner 或 `ros2 run reachability_cli ...` 命令切换到新配置运行

## Git 说明

以下目录默认建议不上传数据内容：

- `tools/data_gen/outputs/`
- `tools/model_validate/inputs/`
- `tools/model_validate/outputs/`

仓库应主要保留：

- 源码
- 配置
- 脚本
- 必要说明文档
