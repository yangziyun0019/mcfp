#!/bin/bash

workspace=$(pwd)
source ~/.bashrc

# CAN
gnome-terminal -t "can" -x sudo bash -c "cd ${workspace};cd ../../.. ; cd ARX_CAN/arx_can; ./arx_can1.sh; exec bash;"
gnome-terminal -t "can" -x sudo bash -c "cd ${workspace};cd ../../.. ; cd ARX_CAN/arx_can; ./arx_can3.sh; exec bash;"
sleep 1

#ARM
gnome-terminal -t "L" -x  bash -c "cd ${workspace}; cd ../../..; cd ROS2/X5_ws; source install/setup.bash && ros2 launch arx_x5_controller v2_pos_control.launch.py; exec bash;"
sleep 0.1
