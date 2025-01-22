#!/bin/bash

# Define the topics to record
topics=(
    "/hpe3d/rgbd_hpe3d" 
    "/hpe3d/openpose_hpe3d"
)

sleep 10
# Start recording with the specified topics
echo "Starting rosbag recording for selected topics..."
rosbag record "${topics[@]}" --duration=30 -O $1.bag


