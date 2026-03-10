#!/bin/bash
# Script: run_validate_zyun.sh
# Purpose: Prepare the server workspace and launch the future model-validation entry point in the current shell environment.
# Usage: bash tools/model_validate/scripts/run_validate_zyun.sh
set -euo pipefail

WORKSPACE="/home/user/Zyun/ros2_workspace"
CFG_VALIDATE="tools/model_validate/configs/robots/aubo/aubo_i5/validate_default.yaml"

cd "$WORKSPACE"

set +u
source "${WORKSPACE}/install/setup.bash"
set -u

if ros2 pkg executables reachability_cli 2>/dev/null | grep -q "network_validation_cli"; then
  ros2 run reachability_cli network_validation_cli --config "${CFG_VALIDATE}"
else
  echo "[todo] network_validation_cli is not implemented yet."
  echo "[plan] ros2 run reachability_cli network_validation_cli --config ${CFG_VALIDATE}"
fi
