#!/bin/bash
# Script: run_validate_local.sh
# Purpose: Prepare the local environment and launch the future model-validation entry point for reachability checks.
# Usage: bash tools/model_validate/scripts/run_validate_local.sh
set -euo pipefail

ROS_DISTRO=humble
WORKSPACE="$HOME/ros2_workspace"
CFG_VALIDATE="tools/model_validate/configs/robots/aubo/aubo_i5/validate_default.yaml"

cd "$WORKSPACE"

if [ -n "${CONDA_SHLVL:-}" ] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda deactivate >/dev/null 2>&1 || true
fi

set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${WORKSPACE}/install/setup.bash"
set -u

if ros2 pkg executables reachability_cli 2>/dev/null | grep -q "network_validation_cli"; then
  ros2 run reachability_cli network_validation_cli --config "${CFG_VALIDATE}"
else
  echo "[todo] network_validation_cli is not implemented yet."
  echo "[plan] ros2 run reachability_cli network_validation_cli --config ${CFG_VALIDATE}"
fi
