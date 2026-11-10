#!/bin/bash
set -e

# Source ROS global installation
source /opt/ros/noetic/setup.bash

# Source the local workspace if it has been built
if [ -f "/root/catkin_ws/devel/setup.bash" ]; then
    source /root/catkin_ws/devel/setup.bash
    echo "Sourced local catkin workspace."
else
    echo "Local workspace not built yet. Run 'catkin build' in /root/catkin_ws"
fi

exec "$@"
