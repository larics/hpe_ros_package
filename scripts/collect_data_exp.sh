#!/bin/bash

# Define the bag file name with timestamp
BAG_NAME="recorded_data_$(date +%Y%m%d_%H%M%S).bag"

# List of topics to record
TOPICS=(
    "/hpe3d/openpose_hpe3d"
    "/red/pose"
    "/red/tracker/input_pose"
    "/control_arm/delta_twist_cmds"
    "/red/position_hold/trajectory"
    "/uav/pose_ref"
    "/uav/test_ref"
    "ctl/viz"
    "ctl/cb_point"
    "ctl/r_hand_normal"
)

echo "Starting rosbag recording..."
rosbag record -O "$BAG_NAME" "${TOPICS[@]}"
