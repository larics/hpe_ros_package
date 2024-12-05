#!/bin/bash

# Define the topics to record
topics=(
    "/hand_2d"
    "/hpe_2d"
    "/hpe_3d"
    "/joint_states"
    "/leftw_point"
    "/pose1"
    "/pose2"
    "/rightw_point"
    "/tf"
    "/tf_static"
    "/vect1"
    "/vect2"
    "/vect3"
    "/vect4"
)

# Start recording with the specified topics
echo "Starting rosbag recording for selected topics..."
rosbag record "${topics[@]}" -O $1.bag


