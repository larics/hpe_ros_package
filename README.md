# hpe_ros_package

ROS package for human pose detection currently. 

Idea is to use it as ROS wrapper for relevant neural networks for 2D and 3D human pose detection. 

Such ROS wrapper could create position or velocity setpoints. 


# TODO: 

- [x] Added median and average filtering for detected wrists
- [ ] Currently only for wrists however, it could be parametrizable 
- [ ] Filtering of whole skeleton will introduce latency
- [ ] Test position control with local coordinate frame 
- [ ] Add hand pose estimation 
- [ ] Add hand gesture estimation
