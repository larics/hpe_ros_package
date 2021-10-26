#!/opt/conda/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from data_to_images import draw_point
from data_to_images import draw_stickman
import cv2

import argparse
import os
import sys
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.inference import get_final_preds, get_max_preds
from utils.utils import create_logger
from utils.transforms import get_affine_transform

from PIL import ImageDraw
from PIL import Image as PILImage

import dataset
import models

from std_msgs.msg import Bool

class HumanPoseEstimationROS():

    def __init__(self, frequency, args):


        rospy.init_node("hpe_simplebaselines")
        
        self.rate = rospy.Rate(frequency)

        # Update configuration file
        update_config(args.cfg)
        reset_config(config, args)
        self.config = config

        # Initialize provided logger
        self.logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'valid')

        # Legacy CUDNN --> probably not necessary 
        cudnn.benchmark = config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = config.CUDNN.ENABLED

        self.model_ready = False
        self.first_img_reciv = False

        self.logger.info("[HPE-SimpleBaselines] Loading model")
        self.model = self._load_model(config)
        self.logger.info("[HPE-SimpleBaselines] Loaded model..")
        self.model_ready = True

        self.multiple_GPUs = False
        if self.multiple_GPUs:
            gpus = [int(i) for i in config.GPUS.split(',')]
            rospy.loginfo("Detected GPUS: {}".format(gpus))
            model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

        # Initialize subscribers/publishers
        self._init_publishers()
        self._init_subscribers()

        self.nn_input = None
        
        # Initialize variables for darknet person bounding boxes
        self.x = None
        self.y = None
        self.w = None
        self.h = None

    def _init_subscribers(self):
        self.camera_sub = rospy.Subscriber("usb_camera/image_raw", Image, self.image_cb, queue_size=1)
        #self.darknet_sub = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.darknet_cb, queue_size=1)

    def _init_publishers(self):
        self.dummy_pub = rospy.Publisher("/dummy_pub", Bool, queue_size=1)
        self.image_pub = rospy.Publisher("/stickman", Image, queue_size=1)
        self.pred_pub = rospy.Publisher("/hpe_preds", Float64MultiArray, queue_size=1)

    def _load_model(self, config):
        
        print("Model name is: {}".format(config.MODEL.NAME))
        model = eval('models.' + config.MODEL.NAME + '.get_pose_net')(
        config, is_train=False)

        self.logger.info("Passed config is: {}".format(config))
        self.logger.info("config.TEST.MODEL.FILE")

        if config.TEST.MODEL_FILE:
            model_state_file = config.TEST.MODEL_FILE
            print('=> loading model from {}'.format(config.TEST.MODEL_FILE))
            model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
        else:
            model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
            print('=> loading model from {}'.format(model_state_file))
            model.load_state_dict(torch.load(model_state_file))           
        
        return model

    def image_cb(self, msg):

        self.first_img_reciv = True

        debug_img = False
        if debug_img:
            self.logger.info("Image width: {}".format(msg.width))
            self.logger.info("Image height: {}".format(msg.height))
            self.logger.info("Data is: {}".format(len(msg.data)))
            self.logger.info("Input shape is: {}".format(input.shape))

        # Transform img to numpy array        
        self.org_img = numpy.frombuffer(msg.data, dtype=numpy.uint8).reshape(msg.height, msg.width, -1)

        # Normalize
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Tensor transformations
        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
        #if self.x != None and self.y != None:
        #    self.cam_img, self.center, self.scale = self.aspect_ratio_scaler(self.org_img, self.x, self.y, self.w, self.h)
        #else:
        #    self.cam_img, self.center, self.scale = self.aspect_ratio_scaler(self.org_img, 0, 0, msg.width, msg.height)

        self.cam_img = cv2.resize(self.org_img, dsize=(348,348), interpolation=cv2.INTER_CUBIC)
        
        #self.logger.info("Scale is: {}".format(self.scale))
        self.center = 384/2, 384/2

        # https://github.com/microsoft/human-pose-estimation.pytorch/issues/26
        
        # Check this scaling
        self.scale = numpy.array([1, 1], dtype=numpy.float32) 
        
        # Transform img to 
        self.nn_input = transform(self.cam_img).unsqueeze(0)   
        
        if debug_img:
            self.logger.info("NN_INPUT {}".format(self.nn_input))                              


    def darknet_cb(self, darknet_boxes):
        
        max_area = 0
        for bbox in darknet_boxes.bounding_boxes: 
            if bbox.Class == "person":
                this_area = (bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin)
                if this_area > max_area:
                    max_area = this_area
                    self.x = bbox.xmin
                    self.y = bbox.ymin
                    self.w = bbox.xmax - bbox.xmin
                    self.h = bbox.ymax - bbox.ymin
        else:
            return


    def aspect_ratio_scaler(self, img, x0, y0, width, height):

        box = [x0, y0, width, height]
        x,y,w,h = box[:4]
        
        center = numpy.zeros((2), dtype=numpy.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        aspect_ratio = width * 1.0 / height
        pixel_std = 200
        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = numpy.array(
            [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
            dtype=numpy.float32)
        if center[0] != -1:
            scale = scale * 1.25


        r = 0
        trans = get_affine_transform(center, scale, r, config.MODEL.IMAGE_SIZE)
        scaled_img = cv2.warpAffine(
            img,
            trans,
            (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        #scaled_img = numpy.array(scaled_img, dtype=numpy.uint8)

        return scaled_img, center, scale

    
    def run(self):

        self.first = True

        while not self.model_ready:
            self.logger.info("SimpleBaselines model for HPE not ready.")
        
        while not rospy.is_shutdown(): 

            if self.first_img_reciv: 
                start_time = rospy.Time.now().to_sec()
                # Convert ROS Image to PIL
                pil_img = PILImage.fromarray(self.org_img.astype('uint8'), 'RGB')
                
                # Get NN Output
                output = self.model(self.nn_input)

                # Heatmaps
                batch_heatmaps = output.cpu().detach().numpy()
                self.logger.info("Heatmap output: {}" +str(batch_heatmaps.shape))

                # Get predictions                
                #preds, maxvals = get_final_preds(config, batch_heatmaps, self.center, self.scale)
                preds, maxvals = get_max_preds(batch_heatmaps)
                duration = rospy.Time.now().to_sec() - start_time
                self.logger.info("[HpeSimpleBaselines] Inference duration is: {}".format(duration))

                # Heatmap size is 88x88, so this scales predictions to image size
                for pred in preds[0]:
                    pred[0] = pred[0]  * (640/88)
                    pred[1] = pred[1]  * (480/88)
                self.logger.info(str(preds[0][0][0]) + "   " + str(preds[0][0][1]))
                
                self.logger.info("Preds are: {}".format(preds))     
                self.logger.info("Preds shape is: {}".format(preds.shape))
                # Preds shape is [1, 16, 2] (or num persons is first dim)
                self.logger.info("Preds shape is: {}".format(preds[0].shape))

                # Draw stickman
                stickman = HumanPoseEstimationROS.draw_stickman(pil_img, preds[0])
                stickman_ros_msg = HumanPoseEstimationROS.convert_pil_to_ros_img(stickman)

                # Prepare predictions for publishing - convert to 1D float list
                converted_preds = []
                for pred in preds[0]:
                    converted_preds.append(pred[0])
                    converted_preds.append(pred[1])
                preds_ros_msg = Float64MultiArray()
                preds_ros_msg.data = converted_preds
                
                self.image_pub.publish(stickman_ros_msg)
                self.pred_pub.publish(preds_ros_msg)
            
            self.rate.sleep()

    @staticmethod
    def draw_stickman(img, predictions):

        draw  = ImageDraw.Draw(img)

        point_r = 2

        for i in range (0, len(predictions)): 
            draw.ellipse([(predictions[i][0] - point_r, predictions[i][1] - point_r), (predictions[i][0] + point_r, predictions[i][1] + point_r)], fill=(255, 0, 0), width=2*point_r)
            if i < len(predictions) - 1 and i != 5 and i != 9:      
                draw.line([(predictions[i][0], predictions[i][1]), (predictions[i + 1][0], predictions[i + 1][1])], fill="green", width=2)


        return img


    @staticmethod
    def convert_pil_to_ros_img(img):
        """Function for converting pillow to ros image
        Args:
            img (PIL.Image.Image): Pillow image that represents GUI
        Returns:
            sensor_msgs.msg._Image.Image: ROS image for image publisher
        """
        img = img.convert('RGB')
        msg = Image()
        stamp = rospy.Time.now()
        msg.height = img.height
        msg.width = img.width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 3 * img.width
        msg.data = numpy.array(img).tobytes()
        return msg   


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file



if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)
    parser.add_argument('--shift-heatmap', 
                        help="shift heatmap", 
                        default=False, 
                        type=bool)

    args, unknown = parser.parse_known_args(sys.argv[2:])
 

    HPERos = HumanPoseEstimationROS(20, args)
    HPERos.run()
