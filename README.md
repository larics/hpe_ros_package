# hpe_ros_package


ROS package for human pose estimation with [Microsoft SimpleBaselines](https://github.com/microsoft/human-pose-estimation.pytorch) algorithm.


### Starting procedure

Start camera, for 3D pose estimation start realsense in realsense docker: 
```
roslaunch realsense2_camera rs_rgbd.launch 
```

### HMI starting procedure 

Start HPE3d: 
``` 
roslaunch hpe_ros_package hpe_3d.launch
```

## Rest of the code for acore and epfl experiments can be found [here](https://github.com/fzoric8/hpe_ros_package). 

Code for the acore and EPFL experiments contains decoupled launch files for easier debugging. 


### TODO High priority: 

 - [ ] Try 3d pose estimation (2D + depth) 
 - [ ] Finish depth control (pitch/depth control)
 - [ ] Publish Cartesian tooltip position 
 - [ ] Implement inverse kinematics

### TODO Low priority: 

 - [ ] Add simple object detection (`darknet_ros`) for better hpe  
 - [ ] Try HRNet --> TBD (not neccesary for now)  
 - [ ] Implement correct post processing as stated [here](https://github.com/microsoft/human-pose-estimation.pytorch/issues/26) 
 
