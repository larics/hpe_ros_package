# Camera HPE with ROS
Download and build docker file from https://github.com/De4d56/hpe_ros
Most things should be built and downloaded by docker
You also need to download pretrained models, you can find links and install instructions at https://github.com/Microsoft/human-pose-estimation.pytorch
To run, enter hpe_ros directory, and then run
``` 
./pose_estimation/run.sh
```
Publish camera images on the /camera topic to get a pose estimation for the image
In order to be able to display images, you might need to disable access control for your display by running:
```
xhost +
```
## Camera simulator
You can also run camera_simulator.py which takes images from a folder and then publishes them for human pose estimation
For instance, you can find some videos seperated in images in this dataset: https://projet.liris.cnrs.fr/voir/activities-dataset/    (Sony camcorder version)
To run, put a video in hpe_ros/pose_estimation/video/video001, and then from pose_estimation run:
```
python camera_simulator.py  video/vid0123/
```
