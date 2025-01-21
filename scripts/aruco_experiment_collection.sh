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

topics=(
    "/hpe3d/rgbd_hpe3d" 
    "/hpe3d/openpose_hpe3d"
    "/aruco_single/pose"
)

sleep 10
# Start recording with the specified topics
echo "STATIONARY RECORDING"
rosbag record "${topics[@]}" --duration=10 -O stationary_aruco.bag

echo "SLOW MOVEMENT ELBOW"
rosbag record "${topics[@]}" --duration=10 -O aruco_distance.bag 



