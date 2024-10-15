#!/usr/bin/python3

import sys
# Adding trt_pose to the sys path
sys.path.append("/root/trt_pose/")

# trt_pose
import trt_pose
import trt_pose.coco
import trt_pose.models
import json

from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

# optimized trt_pose
import torch2trt
from torch2trt import TRTModule

import torch
import rospy

from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo

import torchvision.transforms as transforms
from img_utils import convert_ros_to_pil_img

class TrtPoseROS():

    def __init__(self):

        # Init node
        rospy.init_node("hpe_trt_inference", anonymous=True) 

        self.initialized = False
        self.img_reciv = False
        self.rate = rospy.Rate(100)

        # Init model
        self.model = self._init_model()
        # Init optimized model
        self.model_trt = self._init_optimized_model()

        self._init_subscribers()

        # Init subscribers & pubs     
        self.initialized = True

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.device = torch.device('cuda')

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

    def _init_optimized_model(self): 
        OPTIMIZED_MODEL = '/root/trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
        rospy.loginfo("Optimized model weights loaded succesfuly!")
        return model_trt

    def _init_subscribers(self):
        self.camera_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_cb, queue_size=1)

    def _init_publishers(self):
        pass

    def cinfo_cb(self, msg): 
        pass

    def image_cb(self, msg):
        self.img_reciv = True
        self.pil_img = convert_ros_to_pil_img(msg)

    #https://www.ros.org/news/2018/09/roscon-2017-determinism-in-ros---or-when-things-break-sometimes-and-how-to-fix-it----ingo-lutkebohle.html
    def run(self):

        rospy.loginfo("Entered run method!")
        while not self.initialized: 
            rospy.loginfo("Node is not initialized yet.")
            rospy.Rate(1).sleep()

        if (self.img_reciv and self.initialized):
            rospy.loginfo("Inference loop!")     
            self.nn_input = transforms.functional.to_tensor(self.pil_img).to(self.device)
            self.nn_input.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
            self.nn_in = self.nn_input[None, ...] 
            cmap, paf = self.model_trt(self.nn_in)
            print(cmap, paf)
            self.rate.sleep()    


if __name__ == '__main__':

    trt_ros = TrtPoseROS()
    try:
        while not rospy.is_shutdown(): 
            trt_ros.run()
    except rospy.ROSInterruptException:
        exit()
