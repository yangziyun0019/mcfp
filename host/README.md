# Teleop Host

This folder is now split by responsibility:

- `host/run_teleop.py`
  Fixed input-layer entrypoint. It only reads Arduino + IMU, merges them into a 7D vector, and prints that vector.
- `host/teleop_input.py`
  Shared input module. Arduino reading, IMU reading, waiting logic, and 7D frame assembly live here.
- `host/arx_x5_teleop.py`
  ARX-specific main program. It reuses the shared 7D input layer, verifies ARX comms, applies ARX mapping, and can later send joint targets.
- `host/arx_x5_safe_driver.py`
  ARX diagnostics / mapping driver. It is separate from the input layer on purpose.

## Data flow

1. Arduino Nano sends:
   `CTRL,<pot0_deg>,<pot1_deg>,<pot2_deg>,<grip>`
2. IMU provides:
   `(roll_deg, pitch_deg, yaw_deg)`
3. Shared input layer merges them into:
   `[pot0, pot1, pot2, roll, pitch, yaw, grip]`
4. Robot-specific entrypoints decide what to do with that 7D frame.

## Fixed 7D printer

Use this when you only want to verify that input collection and merging are correct:

```bash
python -m host.run_teleop
```

This script is now fixed-purpose:
- it does not load robot SDK drivers
- it does not contain ARX mapping logic
- it only prints merged 7D vectors in the terminal

If IMU is still offline, it can keep printing with zeroed `roll/pitch/yaw` fallback.

## Shared input options

Both `host/run_teleop.py` and `host/arx_x5_teleop.py` use the same input options from `host/teleop_input.py`.

Edit defaults in:
- `host/teleop_input.py`

Main defaults:
- `DEFAULT_INPUT_CONFIG["arduino_port"]`
- `DEFAULT_INPUT_CONFIG["imu_adapter_port"]`
- `DEFAULT_INPUT_CONFIG["imu_device_prefix"]`
- `DEFAULT_INPUT_CONFIG["imu_device_index"]`
- `DEFAULT_INPUT_CONFIG["allow_imu_zero_fallback"]`

## ARX X5 teleop entrypoint

Use this when you want to run the ARX teleop flow:

```bash
python -m host.arx_x5_teleop
```

Current behavior:
- the script requires a Python 3.10 interpreter so it can load the ARX SDK ABI
- if current `python` is not 3.10, it re-execs into `/usr/bin/python3` by default
- it verifies CAN / ARX SDK communication first
- it waits until both Arduino and IMU streams are ready
- after `Enter`, it prepares the home pose `[90, 15, 15, 0, 0, 0]` with gripper open
- after another `Enter`, it enters teleop mode
- long-hold `m` stops teleop and returns to home
- long-hold `n` returns to `[90, 0, 0, 0, 0, 0]`
- joint targets are smoothed with adaptive filtering plus a small deadband to reduce jitter
- gripper input is interpreted as `1 -> keep opening`, `0 -> keep closing`
- software joint speed limiting is enabled by default for teleop and scripted return moves

Default mode is live:
- it sends joint angles and gripper values to the arm
- use `--dry-run` if you only want to print targets without moving the arm
- use `--joint-speed-limit-deg-per-s 30` to lower the max joint speed cap at launch

Main config lives at the top of:
- `host/arx_x5_teleop.py`

Edit there first:
- `RUN_CONFIG["test_mode"]`
- `RUN_CONFIG["home_joint_deg"]`
- `RUN_CONFIG["home_gripper_value"]`
- `RUN_CONFIG["joint_mapping"]`
- `RUN_CONFIG["teleop_gripper_open_value"]`
- `RUN_CONFIG["teleop_gripper_closed_value"]`
- `RUN_CONFIG["can_port"]`

Recommended Python environment for ARX:

```bash
/usr/bin/python3 -m venv .venv-arx
source .venv-arx/bin/activate
python -m pip install numpy pyserial
ARX_TELEOP_PYTHON=$(pwd)/.venv-arx/bin/python python -m host.arx_x5_teleop
```

Reason:
- ARX SDK is built against Python 3.10 / ROS Humble
- your conda `base` is Python 3.13, which is the wrong ABI for the SDK
- a dedicated `venv` based on `/usr/bin/python3` is a better fit than conda for this script

## ARX X5 keyboard joint test

Use the standalone ARX SDK test before wiring teleop into the arm:

```bash
/usr/bin/python3 host/arx_x5_joint_keyboard.py
```

This script is independent from the 7D input layer. It is only for:
- CAN / SDK connection verification
- joint-by-joint mapping checks
- gripper direction checks

## Files to customize

- `host/teleop_input.py`
  Shared input defaults and 7D merge logic.
- `host/arx_x5_teleop.py`
  ARX-specific teleop flow and joint mapping.
- `host/arx_x5_safe_driver.py`
  ARX mapping, limits, signs, and future actuation logic.
