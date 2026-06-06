#!/bin/bash
cd "$(dirname "$0")"

source /opt/ros/jazzy/setup.bash
source "$HOME/ros2_ws/install/setup.bash"

.venv/bin/python main.py
