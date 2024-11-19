#!/usr/bin/python3

import sys
# Adding trt_pose to the sys path
sys.path.append("/root/trt_pose/")

# trt_pose
import trt_pose
import trt_pose.coco
import trt_pose.models
import json

from trt_pose.draw_objects import DrawPILObjects
from trt_pose.parse_objects import ParseObjects

# optimized trt_pose
import torch2trt
from torch2trt import TRTModule

sys.path.append("/root/trt_pose_hand")

import torch
import rospy
import copy
import cv2
import numpy as np

from std_msgs.msg import Bool, Int64MultiArray
from sensor_msgs.msg import Image as ROSImage
from hpe_ros_msgs.msg import HumanPose2D

import torchvision.transforms as transforms
from img_utils import convert_ros_to_pil_img

from cv_bridge import CvBridge
import traitlets

class HHPoseROS():

    def __init__(self):

        # Init node
        rospy.init_node("hh_trt_inference", anonymous=True, log_level=rospy.DEBUG) 

        self.initialized = False
        self.img_reciv = False
        self.rate = rospy.Rate(100)
        
        # Change camera type # "WEBCAM" or "LUXONIS" or "RS_COMPAT"
        self.camera_type = "RS_COMPAT" 

        # HANDS MODEL
        hands_optim_pth = '/root/trt_pose_hand/model/hand_pose_resnet18_att_244_244_trt.pth'
        # HPE MODEL
        hpe_optim_pth = '/root/trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        self.resize_w, self.resize_h = 224, 224

        # Init hand_model
        self.hand_model_trt = self._init_optimized_model(hands_optim_pth)
        self.hand_topology = self._init_hand_topology()
        # init hpe mode
        self.hpe_model_trt = self._init_optimized_model(hpe_optim_pth)
        self.hpe_topology = self._init_hpe_topology()

        # init subscribers and publishers
        self._init_subscribers()
        self._init_publishers()
        self.initialized = True

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

        # Parse and draw objects to draw skeleton from the camera input 
        self.parse_hpe_objects = ParseObjects(self.hpe_topology)
        self.draw_hpe_objects = DrawPILObjects(self.hpe_topology)
        self.parse_hand_objects = ParseObjects(self.hand_topology, cmap_threshold=0.15, link_threshold=0.15)
        self.draw_hand_objects = DrawPILObjects(self.hand_topology)

        # TODO: Find a way to use this mapping to create message (unify different mappings) --> IMPLEMENTATION THING!
        self.hpe_mapping = {0: "nose", 1: "l_eye", 2: "r_eye", 3: "l_ear",
                            4: "r_ear", 5: "l_shoulder", 6: "r_shoulder", 7: "l_elbow",
                            8: "r_elbow", 9: "l_wrist", 10: "r_wrist", 11: "l_hip",
                            12: "r_hip", 13: "l_knee", 14: "r_knee", 15: "l_ankle", 16: "r_ankle"}

        self.bridge = CvBridge()

    def _init_hand_topology(self): 
        # Load topology
        with open('/root/trt_pose_hand/preprocess/hand_pose.json', 'r') as f: 
            self.hand_pose = json.load(f)
        topology = trt_pose.coco.coco_category_to_topology(self.hand_pose) 
        return topology
    
    def _init_hpe_topology(self):
        # Load topology
        with open('/root/trt_pose/tasks/human_pose/human_pose.json', 'r') as f: 
            self.human_pose = json.load(f)
        topology = trt_pose.coco.coco_category_to_topology(self.human_pose) 
        return topology
      
    def _init_optimized_model(self, optim_model_pth): 
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(optim_model_pth))
        rospy.loginfo("Optimized model weights loaded succesfuly!")
        return model_trt

    def _init_subscribers(self):
        if self.camera_type == "WEBCAM":
            self.camera_sub = rospy.Subscriber("/usb_cam/image_raw", ROSImage, self.image_cb, queue_size=1)
        if self.camera_type == "LUXONIS":
            self.camera_sub = rospy.Subscriber("/oak/rgb/image_raw", ROSImage, self.image_cb, queue_size=1)
        if self.camera_type == "RS_COMPAT": 
            self.camera_sub = rospy.Subscriber("/camera/color/image_raw", ROSImage, self.image_cb, queue_size=1)

    def _init_publishers(self):
        self.image_pub =  rospy.Publisher("/hh_img", ROSImage, queue_size=1)
        self.pred_pub = rospy.Publisher("/hand_keypoint_preds", Int64MultiArray, queue_size=1)
        self.hpe2d_pub = rospy.Publisher("/hpe_2d", HumanPose2D, queue_size=1)

    def image_cb(self, msg):
        rospy.loginfo_once("Recieved oak image!")
        self.pil_img = convert_ros_to_pil_img(msg)
        # Resize PIL image to necessary shape
        self.resized_pil_img = self.pil_img.resize((self.resize_w, self.resize_h))
        self.img_reciv = True

    def create_hpe_msg(self, keypoints):
        # Create ROS msg based on the keypoints
        msg = HumanPose2D()
        msg.header.stamp = rospy.Time.now()
        # TODO: How to make this shorter, based on mapping? 
        msg.nose.x = keypoints[0][0]; msg.nose.y = keypoints[0][1]
        msg.l_eye.x = keypoints[1][0]; msg.l_eye.y = keypoints[1][1]
        msg.r_eye.x = keypoints[2][0]; msg.r_eye.y = keypoints[2][1]
        msg.l_ear.x = keypoints[3][0]; msg.l_ear.y = keypoints[3][1]
        msg.r_ear.x = keypoints[4][0]; msg.r_ear.y = keypoints[4][1]
        msg.l_shoulder.x = keypoints[5][0]; msg.l_shoulder.y = keypoints[5][1]
        msg.r_shoulder.x = keypoints[6][0]; msg.r_shoulder.y = keypoints[6][1]
        msg.l_elbow.x = keypoints[7][0]; msg.l_elbow.y = keypoints[7][1]
        msg.r_elbow.x = keypoints[8][0]; msg.r_elbow.y = keypoints[8][1]
        msg.l_wrist.x = keypoints[9][0]; msg.l_wrist.y = keypoints[9][1]
        msg.r_wrist.x = keypoints[10][0]; msg.r_wrist.y = keypoints[10][1]
        msg.l_hip.x = keypoints[11][0]; msg.l_hip.y = keypoints[11][1]
        msg.r_hip.x = keypoints[12][0]; msg.r_hip.y = keypoints[12][1]
        msg.l_knee.x = keypoints[13][0]; msg.l_knee.y = keypoints[13][1]
        msg.r_knee.x = keypoints[14][0]; msg.r_knee.y = keypoints[14][1]
        msg.l_ankle.x = keypoints[15][0]; msg.l_ankle.y = keypoints[15][1]
        return msg

    def pub_hpe_pred(self, msg): 
        self.hpe_2d_pub.publish(msg)

    # TODO: Both predict methods could be reduced to one method
    def predict_hpe(self, nn_in):
        cmap, paf = self.hpe_model_trt(nn_in)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_hpe_objects(cmap, paf)
        return counts, objects, peaks
    
    def predict_hands(self, nn_in):
        cmap, paf = self.hand_model_trt(nn_in)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_hand_objects(cmap, paf)
        return counts, objects, peaks

    #https://www.ros.org/news/2018/09/roscon-2017-determinism-in-ros---or-when-things-break-sometimes-and-how-to-fix-it----ingo-lutkebohle.html
    def run(self):

        while not self.initialized: 
            rospy.loginfo("Node is not initialized yet.")
            rospy.Rate(1).sleep()

        if (self.img_reciv and self.initialized):
            rospy.loginfo_throttle_identical(10, "Inference loop!")     
            # TODO: Check duration of the copy operation [maybe slows things down]
            # Doesn't work without resizing
            # TODO: Measure duration --> inference duration is not slow 
            # TODO: Check why it detects only one hand
            # TODO: Speed it up
            self.start_time = rospy.Time.now().to_sec()
            self.inf_img = copy.deepcopy(self.resized_pil_img)
            self.nn_input = transforms.functional.to_tensor(self.inf_img).to(torch.device('cuda'))
            self.nn_input.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
            self.nn_in = self.nn_input[None, ...] 
            # Predictions of the HPE and the hand predictions
            hpe_counts, hpe_objects, hpe_peaks = self.predict_hpe(self.nn_in)
            hand_counts, hand_objects, hand_peaks = self.predict_hands(self.nn_in)

            # TODO: Modify both draw_methods to use this data to extract 3D keypoints for them to be published
            # TODO: How to specify ROS message to use this data and how to convert this data to 3D keypoints? 
            img, hpe_keypoints = self.draw_hpe_objects(self.inf_img, hpe_counts, hpe_objects, hpe_peaks)
            img, hand_keypoints = self.draw_hand_objects(img, hand_counts, hand_objects, hand_peaks)
            # Publish predictions on ROS topic
            hpe_msg = self.create_hpe_msg(hpe_keypoints["0"])
            self.hpe2d_pub.publish(hpe_msg)
            # Keypoints are: 1st detection, 2nd detection, 3rd detection, ...
            self.image_pub.publish((self.bridge.cv2_to_imgmsg(img, 'rgb8')))
            self.end_time = rospy.Time.now().to_sec()
            #rospy.logdebug("Inference duration is: {}".format(self.end_time - self.start_time))
            # Inference duration is 0.05 --> 20 Hz is inference of the HPE + hands
            #self.rate.sleep()    

if __name__ == '__main__':

    trt_ros = HHPoseROS()
    try:
        while not rospy.is_shutdown(): 
            trt_ros.run()
    except rospy.ROSInterruptException:
        exit()
