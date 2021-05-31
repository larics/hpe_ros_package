# hpe_ros_package


ROS package for human pose estimation with [Microsoft SimpleBaselines](https://github.com/microsoft/human-pose-estimation.pytorch) algorithm.

Input to the network should be cropped image of a person (detected person). 


You need to download weights for COCO or MPII dataset from following [link](https://onedrive.live.com/?authkey=%21AKqtqKs162Z5W7g&id=56B9F9C97F261712%2110692&cid=56B9F9C97F261712). 

You need [usb-cam](https://github.com/ros-drivers/usb_cam) ROS package for testing it on your PC with webcam. 

### Starting procedure

Launch your cam, in this case web-cam: 
```
roslaunch usb_cam usb_cam-test.launch 
``` 
After downloading weights to `/home/developer/catkin_ws/src/hpe_ros_package/src/lib/models` you can run your HPE inference 
with following: 
```
roslaunch hpe webcam_inference.launch
```

TODO: 
 - [ ] Add simple object detection (`darknet_ros`) for better pose estimate 
 - [ ] Add topic for publishing predictions 
 - [ ] Implement correct post processing as stated [here](https://github.com/microsoft/human-pose-estimation.pytorch/issues/26) 
 
