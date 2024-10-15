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

from std_msgs.msg import Bool
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

        # Init model
        self.model = self._init_model()
        # Init optimized model
        self.model_trt = self._init_optimized_model()

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

    def _init_model(self): 
        # Load topology
        with open('/root/trt_pose/tasks/human_pose/human_pose.json', 'r') as f: 
            human_pose = json.load(f)

        # Load topology
        self.topology = trt_pose.coco.coco_category_to_topology(human_pose) 

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
        self.camera_sub = rospy.Subscriber("/usb_cam/image_raw", ROSImage, self.image_cb, queue_size=1)

    def _init_publishers(self):
        self.image_pub =  rospy.Publisher("/person_img", ROSImage, queue_size=1)

    def cinfo_cb(self, msg): 
        pass

    def image_cb(self, msg):
        self.pil_img = convert_ros_to_pil_img(msg)
        # Resize PIL image to necessary shape
        self.resized_pil_img = self.pil_img.resize((224, 224))
        self.img_reciv = True

    #https://www.ros.org/news/2018/09/roscon-2017-determinism-in-ros---or-when-things-break-sometimes-and-how-to-fix-it----ingo-lutkebohle.html
    def run(self):

        rospy.loginfo("Entered run method!")
        while not self.initialized: 
            rospy.loginfo("Node is not initialized yet.")
            rospy.Rate(1).sleep()

        if (self.img_reciv and self.initialized):
            rospy.loginfo("Inference loop!")     
            # TODO: Check duration of the copy operation [maybe slows things down]
            self.inf_img = copy.deepcopy(self.resized_pil_img)
            self.nn_input = transforms.functional.to_tensor(self.inf_img).to(self.device)
            self.nn_input.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
            self.nn_in = self.nn_input[None, ...] 
            cmap, paf = self.model_trt(self.nn_in)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = self.parse_objects(cmap, paf)
            print("Counts: ", counts)

            self.inf_img = self.draw_objects(self.inf_img, counts, objects, peaks)
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
