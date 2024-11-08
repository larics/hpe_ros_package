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

import torch
import rospy
import copy

from std_msgs.msg import Bool, Int64MultiArray
from sensor_msgs.msg import Image as ROSImage

import torchvision.transforms as transforms
from img_utils import convert_ros_to_pil_img, convert_pil_to_ros_img

from cv_bridge import CvBridge

class TrtPoseROS():

    def __init__(self):

        # Init node
        rospy.init_node("hpe_trt_inference", anonymous=True) 

        self.initialized = False
        self.img_reciv = False
        self.rate = rospy.Rate(100)
        
        # Configure model pths
        
        MODEL = "resnet18_baseline_att_224x224_A"
        #MODEL = "densenet121_baseline_att_256x256_B"
        # Change camera type 
        self.camera_type = "RS_COMPAT" # "WEBCAM" or "LUXONIS" or "RS_COMPAT"

        if MODEL == "resnet18_baseline_att_224x224_A": 
            weights_pth = '/root/trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249.pth'
            optim_pth = '/root/trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
            self.resize_w, self.resize_h = 224, 224
        if MODEL == "densenet121_baseline_att_256x256_B":
            weights_pth = '/root/trt_pose/tasks/human_pose/densenet121_baseline_att_256x256_B_epoch_160.pth'
            optim_pth = '/root/trt_pose/tasks/human_pose/densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
            self.resize_w, self.resize_h = 224, 224

        # Init model
        self.model = self._init_model(MODEL, weights_pth)
        # Init optimized model
        self.model_trt = self._init_optimized_model(optim_pth)

        self._init_subscribers()
        self._init_publishers()

        # Init subscribers & pubs     
        self.initialized = True

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.device = torch.device('cuda')
        
        # Parse and draw objects to draw skeleton from the camera input 
        self.parse_objects = ParseObjects(self.topology)
        self.draw_objects = DrawPILObjects(self.topology)

        self.bridge = CvBridge()

    def _init_model(self, model_name, model_pth): 
        # Load topology
        with open('/root/trt_pose/tasks/human_pose/human_pose.json', 'r') as f: 
            human_pose = json.load(f)

        # Load topology
        self.topology = trt_pose.coco.coco_category_to_topology(human_pose) 

        num_parts = len(human_pose['keypoints'])
        num_links = len(human_pose['skeleton'])
        
        # Load model
        if model_name == "resnet18_baseline_att_224x224_A": 
            model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
        if model_name == "densenet121_baseline_att_256x256_B":
            model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()

        model.load_state_dict(torch.load(model_pth))
        rospy.loginfo("Regular model weights loaded succesfuly!")
        return model

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
        self.image_pub =  rospy.Publisher("/person_img", ROSImage, queue_size=1)
        self.pred_pub = rospy.Publisher("/hpe_preds", Int64MultiArray, queue_size=1)

    def cinfo_cb(self, msg): 
        pass

    def image_cb(self, msg):
        rospy.loginfo_once("Recieved oak image!")
        self.pil_img = convert_ros_to_pil_img(msg)
        # Resize PIL image to necessary shape
        self.resized_pil_img = self.pil_img.resize((self.resize_w, self.resize_h))
        self.img_reciv = True

    def publish_predictions(self, keypoints):
        # Simple predictions publisher (publish detected pixels just as a test)
        msg = Int64MultiArray()
        for k in keypoints:
            msg.data.append(k[0])
            msg.data.append(k[1])  
        self.pred_pub.publish(msg)

    #https://www.ros.org/news/2018/09/roscon-2017-determinism-in-ros---or-when-things-break-sometimes-and-how-to-fix-it----ingo-lutkebohle.html
    def run(self):

        while not self.initialized: 
            rospy.loginfo("Node is not initialized yet.")
            rospy.Rate(1).sleep()

        if (self.img_reciv and self.initialized):
            rospy.loginfo_throttle_identical(10, "Inference loop!")     
            # TODO: Check duration of the copy operation [maybe slows things down]
            # Doesn't work without resizing
            self.inf_img = copy.deepcopy(self.resized_pil_img)
            self.nn_input = transforms.functional.to_tensor(self.inf_img).to(self.device)
            self.nn_input.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
            self.nn_in = self.nn_input[None, ...] 
            cmap, paf = self.model_trt(self.nn_in)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = self.parse_objects(cmap, paf)

            self.inf_img, keypoints = self.draw_objects(self.inf_img, counts, objects, peaks)
            # Publish predictions on ROS topic
            self.publish_predictions(keypoints)
            #img_msg = convert_pil_to_ros_img(self.inf_img)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.inf_img, 'rgb8'))

            self.rate.sleep()    


if __name__ == '__main__':

    trt_ros = TrtPoseROS()
    try:
        while not rospy.is_shutdown(): 
            trt_ros.run()
    except rospy.ROSInterruptException:
        exit()
