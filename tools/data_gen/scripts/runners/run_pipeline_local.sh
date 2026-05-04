#!/bin/bash
# Script: run_pipeline_local.sh
# Purpose: Run a local reachability dataset pipeline for Aubo, RealMan, or Franka.
# Usage: bash tools/data_gen/scripts/runners/run_pipeline_local.sh [all|pos|orient|vis]
set -euo pipefail

# ---------- 用户可按需修改 ----------
ROS_DISTRO="${ROS_DISTRO:-humble}"
WORKSPACE="${WORKSPACE:-$HOME/ros2_workspace}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-reachability}"
ROBOT="${ROBOT:-aubo}"  # aubo | realman | franka

case "${ROBOT}" in
  aubo|aubo_i5)
    DEFAULT_BUILD_PKGS="aubo_description aubo_moveit_config reachability_cli"
    DEFAULT_CFG_POS="tools/data_gen/configs/robots/aubo/aubo_i5/position_3mm.yaml"
    DEFAULT_CFG_ORIENT="tools/data_gen/configs/robots/aubo/aubo_i5/orientation_3mm.yaml"
    DEFAULT_OUTPUT_DIR="tools/data_gen/outputs/aubo/aubo_i5/voxel_3mm"
    DEFAULT_LOG_PREFIX="aubo_i5"
    ;;
  realman|rm65)
    DEFAULT_BUILD_PKGS="rm_description rm_moveit_config reachability_cli"
    DEFAULT_CFG_POS="tools/data_gen/configs/robots/realman/rm65/position_3mm.yaml"
    DEFAULT_CFG_ORIENT="tools/data_gen/configs/robots/realman/rm65/orientation_3mm.yaml"
    DEFAULT_OUTPUT_DIR="tools/data_gen/outputs/realman/rm65/voxel_3mm"
    DEFAULT_LOG_PREFIX="rm65"
    ;;
  franka|panda)
    DEFAULT_BUILD_PKGS="franka_emika_panda franka_emika_panda_moveit_config reachability_cli"
    DEFAULT_CFG_POS="tools/data_gen/configs/robots/franka_emika_panda/panda/position_3mm.yaml"
    DEFAULT_CFG_ORIENT="tools/data_gen/configs/robots/franka_emika_panda/panda/orientation_3mm.yaml"
    DEFAULT_OUTPUT_DIR="tools/data_gen/outputs/franka_emika_panda/panda/voxel_3mm"
    DEFAULT_LOG_PREFIX="franka_panda"
    ;;
  *)
    echo "Unknown ROBOT='${ROBOT}'. Use ROBOT=aubo, ROBOT=realman, or ROBOT=franka." >&2
    exit 1
    ;;
esac

# 仅构建与数据生成相关的包（更快）
# Description / MoveIt config 包也要一起安装进 overlay，package:// mesh 才能正确解析。
BUILD_PKGS="${BUILD_PKGS:-${DEFAULT_BUILD_PKGS}}"
PKG_NAME="reachability_cli"
PKG_SRC_DIR="${WORKSPACE}/src/core/reachability_cli"
PKG_BUILD_DIR="${WORKSPACE}/build/${PKG_NAME}"
PKG_INSTALL_DIR="${WORKSPACE}/install/${PKG_NAME}"
PKG_CACHE_FILE="${PKG_BUILD_DIR}/CMakeCache.txt"

# 正式 3mm 位置数据与 V1.3 1024-anchor 姿态数据配置。
CFG_POS="${CFG_POS:-${DEFAULT_CFG_POS}}"
CFG_ORIENT="${CFG_ORIENT:-${DEFAULT_CFG_ORIENT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
LOG_PREFIX="${LOG_PREFIX:-${DEFAULT_LOG_PREFIX}}"

# 是否清理旧构建产物
CLEAN_BUILD="${CLEAN_BUILD:-false}"
BUILD_BEFORE_RUN="${BUILD_BEFORE_RUN:-true}"

# 是否在生成结束后运行可视化
RUN_VIS="${RUN_VIS:-false}"

# 32 线程默认吃满本机 CPU；需要改线程时可以在命令前覆盖 OMP_NUM_THREADS。
OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
OMP_DYNAMIC="${OMP_DYNAMIC:-false}"
OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
OMP_PLACES="${OMP_PLACES:-cores}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
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
    echo "  ROBOT=aubo|realman|franka selects robot configs; default ROBOT=aubo"
    echo "  all    default, run position first and then orientation"
    echo "  pos    run only position SDF mining"
    echo "  orient run only orientation SDF mining; requires ${OUTPUT_DIR}/dataset.h5"
    echo "  vis    run only visualization scripts"
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
if [ "${BUILD_BEFORE_RUN}" = true ]; then
  colcon build --packages-select ${BUILD_PKGS} --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
fi

# 5) 激活工作空间环境
set +u
source "${WORKSPACE}/install/setup.bash"
set -u

export OMP_NUM_THREADS
export OMP_DYNAMIC
export OMP_PROC_BIND
export OMP_PLACES
export MKL_NUM_THREADS
export OPENBLAS_NUM_THREADS
mkdir -p "${LOG_DIR}"

echo "[cmrdf] workspace: ${WORKSPACE}"
echo "[cmrdf] robot: ${ROBOT}"
echo "[cmrdf] mode: ${MODE}"
echo "[cmrdf] position config: ${CFG_POS}"
echo "[cmrdf] orientation config: ${CFG_ORIENT}"
echo "[cmrdf] output dir: ${OUTPUT_DIR}"
echo "[cmrdf] logs: ${LOG_DIR}"
echo "[cmrdf] OMP_NUM_THREADS=${OMP_NUM_THREADS} OMP_PROC_BIND=${OMP_PROC_BIND} OMP_PLACES=${OMP_PLACES}"

run_position() {
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="${LOG_DIR}/${LOG_PREFIX}_position_3mm_${stamp}.log"
  echo "[cmrdf] position SDF start: $(date)"
  /usr/bin/time -v ros2 run reachability_cli dataset_generator_cli \
    --config "${CFG_POS}" \
    2>&1 | tee "${log_file}"
  echo "[cmrdf] position SDF done: $(date)"
  echo "[cmrdf] position log: ${log_file}"
}

run_orientation() {
  local base_h5="${OUTPUT_DIR}/dataset.h5"
  if [ ! -f "${base_h5}" ]; then
    echo "[error] Missing ${base_h5}. Run position first: bash $0 pos" >&2
    exit 2
  fi
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local log_file="${LOG_DIR}/${LOG_PREFIX}_orientation_3mm_1024_${stamp}.log"
  echo "[cmrdf] orientation SDF start: $(date)"
  /usr/bin/time -v ros2 run reachability_cli orientation_dataset_cli \
    --config "${CFG_ORIENT}" \
    2>&1 | tee "${log_file}"
  echo "[cmrdf] orientation SDF done: $(date)"
  echo "[cmrdf] orientation log: ${log_file}"
}

# 生成后的可视化脚本依赖 NumPy / HDF5 / Matplotlib，优先使用专门的 Conda 环境。
USE_CONDA_VIS=false
if [ "${RUN_VIS}" = true ] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  if conda run -n "${CONDA_ENV_NAME}" python -c "import h5py, matplotlib, numpy" >/dev/null 2>&1; then
    echo "[info] Using Conda env ${CONDA_ENV_NAME} for visualization scripts"
    USE_CONDA_VIS=true
  fi
fi

run_vis_python() {
  if [ "${USE_CONDA_VIS}" = true ]; then
    conda run --no-capture-output -n "${CONDA_ENV_NAME}" python "$@"
  else
    python3 "$@"
  fi
}

# 6) 位置采样（生成 dataset.h5）
if [ "${RUN_POS}" = true ]; then
  run_position
fi

# 7) 姿态采样（生成 dataset_orient.h5）
if [ "${RUN_ORIENT}" = true ]; then
  run_orientation
fi

# 8) 可视化（按需开启）
if [ "${RUN_VIS}" = true ]; then
  run_vis_python tools/data_gen/scripts/visualize/plot_occupancy_voxels.py
  run_vis_python tools/data_gen/scripts/visualize/plot_orient_samples.py
  run_vis_python tools/data_gen/scripts/visualize/plot_anchor_distribution.py
fi
