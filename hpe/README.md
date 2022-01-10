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

### HMI starting procedure 

Starting procedure for HMI is to setup `ROS_MASTER_URI` and `ROS_IP` to the values of host PC to enable 
streaming to AR glasses. 
After that we can run following script: 

```
roslaunch hpe hmi_integration.launch 
```


### TODO High priority: 

 - [ ] Move to compressed image to minimize latency for AR glasses 
 - [ ] Add different zones configuration 
 - [ ] Find accuracy metrics for human pose estimation  

### TODO Low priority: 

 - [ ] Try 3d pose estimation 
 - [ ] Add simple object detection (`darknet_ros`) for better pose estimate 
 - [ ] Implement correct post processing as stated [here](https://github.com/microsoft/human-pose-estimation.pytorch/issues/26) 
 
