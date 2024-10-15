#!/usr/bin/python3

import sys
print("SYS executable is: ")
print(sys.executable)
print("SYS path is: ")
# Adding trt_pose to the sys path
sys.path.append("/root/trt_pose/")

import trt_pose
import trt_pose.coco
import trt_pose.models
import json

import torch
import rospy

from std_msgs.msg import Bool

class TrtPoseROS():

    def __init__(self):

        # Init node
        rospy.init_node("hpe_trt_inference", anonymous=True) 

        self.initialized = False
        self.rate = rospy.Rate(100)

        # Init model
        self.model = self._init_model()

        # Init subscribers & pubs     

        self.initialized = True

    def _init_model(self): 
        
        # Load topology
        with open('/root/trt_pose/tasks/human_pose/human_pose.json', 'r') as f: 
            human_pose = json.load(f)

        num_parts = len(human_pose['keypoints'])
        num_links = len(human_pose['skeleton'])
        
        # Load model 
        model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
        MODEL_WEIGHTS = '/root/trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249.pth'
        model.load_state_dict(torch.load(MODEL_WEIGHTS))
        
        rospy.loginfo("Regular model weights loaded succesfuly!")
        return model

    def _init_subscribers(self):
        
        if self.use_depth:
            # Xtion Cam
            self.camera_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_cb, queue_size=1)
        else:
            # USB Cam
            self.camera_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_cb, queue_size=1)
            self.camera_info_sub = rospy.Subscriber("camera/color/camera_info", CameraInfo, self.cinfo_cb, queue_size=1)
            
        #self.darknet_sub = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.darknet_cb, queue_size=1)

    def _init_publishers(self):
        pass

    def cinfo_cb(self, msg): 
        pass

    def image_cb(self, msg):
        pass

    #https://www.ros.org/news/2018/09/roscon-2017-determinism-in-ros---or-when-things-break-sometimes-and-how-to-fix-it----ingo-lutkebohle.html
    def run(self):

        while not self.initialized: 
            rospy.loginfo("Node is not initialized yet.")
            rospy.Rate(1).sleep()


        if (self.first_img_reciv and self.nn_input_formed):
            rospy.loginfo("Inference loop!")      
            self.rate.sleep()    


if __name__ == '__main__':

    trt_ros = TrtPoseROS()
    try:
        while not rospy.is_shutdown(): 
            rospy.spin()
            TrtPoseROS.run()
    except rospy.ROSInterruptException:
        exit()
