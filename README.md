# hpe_ros_package

ROS package for human pose detection currently. 

Idea is to use it as ROS wrapper for relevant neural networks for 2D and 3D human pose detection. 

Such ROS wrapper could create position or velocity setpoints. 


# TODO: 

- [x] Added median and average filtering for detected wrists
- [ ] Currently only for wrists however, it could be parametrizable
- [ ] Test Microsoft HRNet/greek NN and compare speeds 
- [ ] Test position control with local coordinate frame 
- [ ] Add hand pose estimation 
- [ ] Add hand gesture estimation (classification) --> mode change
- [ ] Test different input modalities (2D/3D) and choose ideal one for robotic arm teleoperation

# low-priority todo: 

- [ ] Measure latency based on filtering window size (system reactivity is reduced with bigger window size) 
- [ ] Filter whole skeleton
- [ ] Remove for loops 

