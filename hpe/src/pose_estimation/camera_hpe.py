#!/usr/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import rospy
import data_to_images
from sensor_msgs.msg import Image
from data_to_images import draw_point
import cv2

import argparse
import os
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
from core.loss import JointsMSELoss
from core.function import validate
from core.function import network_use
from utils.utils import create_logger
from utils.transforms import get_affine_transform

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

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
    parser.add_argument('--flip-test',
                        help='use flip test',
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file


def main(data):
    org_img = numpy.frombuffer(data.data, dtype=numpy.uint8).reshape(data.height, data.width, -1)
    #img = torch.from_numpy(org_img)

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    #transposed_img = numpy.transpose(img, (2,0,1))
    #transposed_img = transposed_img.type(torch.FloatTensor)
    #valid_loader = transposed_img[None,:,:,:]

    box = [0, 0, 720, 576]
    x,y,w,h = box[:4]

    center = numpy.zeros((2), dtype=numpy.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    aspect_ratio = data.width * 1.0 / data.height
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
    input = cv2.warpAffine(
        org_img,
        trans,
        (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input = transform(input).unsqueeze(0)

    # Calculate metadata
    #center = numpy.array([data.height / 2, data.width / 2])
    #scale = numpy.array([144 / data.height, 188 / data.width])
    #print(center, scale)

    # use network for output
    preds = network_use(config, input, model, criterion, center, scale)

    #start_point = (150, 50)
    #end_point = (150 + 500, 50 + 450)
    #color = (255,0,0)
    #cv2.rectangle(org_img, start_point, end_point, color,3)
    draw_point(preds, org_img)

if __name__ == '__main__':
    args = parse_args()
    reset_config(config, args)
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.' + config.MODEL.NAME + '.get_pose_net')(
        config, is_train=False
    )

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()


    rospy.init_node("camera_hpe")
    camera_subscriber = rospy.Subscriber("usb_cam/image_raw", Image, main, queue_size=1)
    rospy.spin()
