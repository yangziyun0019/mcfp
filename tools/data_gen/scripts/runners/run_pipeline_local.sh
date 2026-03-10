#!/bin/bash
# Script: run_pipeline_local.sh
# Purpose: Run the local reachability dataset pipeline for position sampling, orientation sampling, or visualization.
# Usage: bash tools/data_gen/scripts/runners/run_pipeline_local.sh [all|pos|orient|vis]
set -euo pipefail

# ---------- 用户可按需修改 ----------
ROS_DISTRO=humble
WORKSPACE="$HOME/ros2_workspace"

# 仅构建与数据生成相关的包（更快）
BUILD_PKGS="reachability_cli"
PKG_NAME="reachability_cli"
PKG_SRC_DIR="${WORKSPACE}/src/core/reachability_cli"
PKG_BUILD_DIR="${WORKSPACE}/build/${PKG_NAME}"
PKG_INSTALL_DIR="${WORKSPACE}/install/${PKG_NAME}"
PKG_CACHE_FILE="${PKG_BUILD_DIR}/CMakeCache.txt"

# 切换机器人时，优先修改这两个配置路径
# 位置采样（生成 dataset.h5）配置
CFG_POS="tools/data_gen/configs/robots/aubo/aubo_i5/position_3mm.yaml"

# 姿态采样（生成 dataset_orient.h5）配置
CFG_ORIENT="tools/data_gen/configs/robots/aubo/aubo_i5/orientation_3mm.yaml"

# 是否清理旧构建产物
CLEAN_BUILD=false

# 是否在生成结束后运行可视化
RUN_VIS=false
# -----------------------------------

MODE="${1:-all}"  # all | pos | orient | vis

case "${MODE}" in
  all)
    RUN_POS=true
    RUN_ORIENT=true
    ;;
  pos)
    RUN_POS=true
    RUN_ORIENT=false
    ;;
  orient)
    RUN_POS=false
    RUN_ORIENT=true
    ;;
  vis)
    RUN_POS=false
    RUN_ORIENT=false
    RUN_VIS=true
    ;;
  *)
    echo "Usage: bash $0 [all|pos|orient|vis]"
    exit 1
    ;;
esac

# 1) 进入工作空间
cd "$WORKSPACE"

# Try to leave the current Conda env without failing in a non-initialized shell.
if [ -n "${CONDA_SHLVL:-}" ] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda deactivate >/dev/null 2>&1 || true
fi

# 2) 激活 ROS 2 全局环境
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
set -u

# 3) 可选清理构建目录
if [ "$CLEAN_BUILD" = true ]; then
  rm -rf build/ install/ log/
fi

# Clear stale per-package CMake cache after package relocation.
if [ -f "${PKG_CACHE_FILE}" ]; then
  if grep -q "src/reachability_cli" "${PKG_CACHE_FILE}" && ! grep -q "${PKG_SRC_DIR}" "${PKG_CACHE_FILE}"; then
    echo "[info] Detected stale CMake cache for ${PKG_NAME}; clearing ${PKG_BUILD_DIR} and ${PKG_INSTALL_DIR}"
    rm -rf "${PKG_BUILD_DIR}" "${PKG_INSTALL_DIR}"
  fi
fi

# 4) 编译需要的包
colcon build --packages-select ${BUILD_PKGS} --symlink-install

# 5) 激活工作空间环境
set +u
source "${WORKSPACE}/install/setup.bash"
set -u

# 6) 位置采样（生成 dataset.h5）
if [ "${RUN_POS}" = true ]; then
  ros2 run reachability_cli dataset_generator_cli --config "${CFG_POS}"
fi

# 7) 姿态采样（生成 dataset_orient.h5）
if [ "${RUN_ORIENT}" = true ]; then
  ros2 run reachability_cli orientation_dataset_cli --config "${CFG_ORIENT}"
fi

# 8) 可视化（按需开启）
if [ "${RUN_VIS}" = true ]; then
  python3 tools/data_gen/scripts/visualize/plot_occupancy_voxels.py
  python3 tools/data_gen/scripts/visualize/plot_orient_samples.py
  python3 tools/data_gen/scripts/visualize/plot_anchor_distribution.py
fi
