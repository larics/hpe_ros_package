#!/bin/bash

# Define the topics to record
topics=(
    "/hand_2d"
    "/hh_img"
    "/hpe_2d"
    "/joint_states"
    "/leftw_point"
    "/pose1"
    "/pose2"
    "/rectify_color/parameter_descriptions"
    "/rectify_color/parameter_updates"
    "/rightw_point"
    "/rosout"
    "/rosout_agg"
    "/tf"
    "/tf_static"
    "/upper_body_3d"
    "/vect1"
    "/vect2"
    "/vect3"
    "/vect4"
)

# Start recording with the specified topics
echo "Starting rosbag recording for selected topics..."
rosbag record "${topics[@]}" -O topic2.bag


