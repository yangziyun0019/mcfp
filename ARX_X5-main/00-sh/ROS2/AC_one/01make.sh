#!/bin/bash
source ~/.bashrc
workspace=$(pwd)

#AC one
gnome-terminal -t "L" -x  bash -c "cd ${workspace}; cd ../../..; cd ROS2/X5_ws; rm -rf build install log .catkin_workspace src/CMakeLists.txt;colcon build; exec bash;"
sleep 0.5
#joy
gnome-terminal -t "L" -x  bash -c "cd ${workspace}; cd ../../..; cd arx_joy; rm -rf build install log .catkin_workspace src/CMakeLists.txt;colcon build; exec bash;"
sleep 0.5


#VR
#gnome-terminal -t "vr" -x  bash -c "cd ${workspace}; cd ../../..; cd ARX_VR_SDK/ROS2; rm -rf build install log .catkin_workspace;./port.sh; exec bash;"
