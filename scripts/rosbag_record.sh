#!/bin/bash 

bash -c "source /opt/ros/melodic/setup.bash; rosbag record /upper_body_3d /right_arm /left_arm /cart_right_arm /cart_left_arm"


