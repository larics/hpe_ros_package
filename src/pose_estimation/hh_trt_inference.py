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
from preprocessdata import preprocessdata

import torch
import rospy
import copy
import cv2
import numpy as np

from std_msgs.msg import Bool, Int64MultiArray
from sensor_msgs.msg import Image as ROSImage

import torchvision.transforms as transforms
from img_utils import convert_ros_to_pil_img, convert_pil_to_ros_img

from cv_bridge import CvBridge
import traitlets

class HHPoseROS():

    def __init__(self):

        # Init node
        rospy.init_node("hh_trt_inference", anonymous=True) 

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

        self._init_subscribers()
        self._init_publishers()

        # Init subscribers & pubs     
        self.initialized = True

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.device = torch.device('cuda')
        
        # Parse and draw objects to draw skeleton from the camera input 
        self.parse_hpe_objects = ParseObjects(self.hpe_topology)
        self.draw_hpe_objects = DrawPILObjects(self.hpe_topology)
        self.parse_hand_objects = ParseObjects(self.hand_topology, cmap_threshold=0.15, link_threshold=0.15)
        self.draw_hand_objects = DrawPILObjects(self.hand_topology)

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


    def image_cb(self, msg):
        rospy.loginfo_once("Recieved oak image!")
        self.pil_img = convert_ros_to_pil_img(msg)
        # Resize PIL image to necessary shape
        self.resized_pil_img = self.pil_img.resize((self.resize_w, self.resize_h))
        self.img_reciv = True

    # TODO: Both predict methods could be reduced to one method
    def predict_hpe(self, nn_in):
        cmap, paf = self.hpe_model_trt(nn_in)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_hpe_objects(cmap, paf)
        return counts, objects, peaks
    
    def predict_hand(self, nn_in):
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
            self.nn_input = transforms.functional.to_tensor(self.inf_img).to(self.device)
            self.nn_input.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
            self.nn_in = self.nn_input[None, ...] 
            hpe_counts, hpe_objects, hpe_peaks = self.predict_hpe(self.nn_in)
            hand_counts, hand_objects, hand_peaks = self.predict_hand(self.nn_in)
            img, keypoints = self.draw_hpe_objects(self.inf_img, hpe_counts, hpe_objects, hpe_peaks)
            img, keypoints = self.draw_hand_objects(img, hand_counts, hand_objects, hand_peaks)
            self.image_pub.publish((self.bridge.cv2_to_imgmsg(img, 'rgb8')))
            self.end_time = rospy.Time.now().to_sec()
            rospy.loginfo("Inference duration is: {}".format(self.end_time - self.start_time))
            #self.rate.sleep()    

if __name__ == '__main__':

    trt_ros = HHPoseROS()
    try:
        while not rospy.is_shutdown(): 
            trt_ros.run()
    except rospy.ROSInterruptException:
        exit()
