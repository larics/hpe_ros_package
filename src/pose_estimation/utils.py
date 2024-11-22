#!/usr/bin/python3

from hpe_ros_msgs.msg import HumanPose2D, HandPose2D
import numpy as np

def limitCmd(cmd, upperLimit, lowerLimit):
    if cmd > upperLimit: 
        cmd = upperLimit
    if cmd < lowerLimit: 
        cmd = lowerLimit
    return cmd

def arrayToVect(array, vect): 
    vect.x = array[0]
    vect.y = array[1]
    vect.z = array[2]
    return vect

# Create Rotation matrices
def get_RotX(angle):  
    
    RX = np.array([[1, 0, 0], 
                   [0, np.cos(angle), -np.sin(angle)], 
                   [0, np.sin(angle), np.cos(angle)]])
    
    return RX

def get_RotY(angle): 
    
    RY = np.array([[np.cos(angle), 0, np.sin(angle)], 
                   [0, 1, 0], 
                   [-np.sin(angle), 0, np.cos(angle)]])
    return RY
    
def get_RotZ(angle): 
    
    RZ = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0], 
                   [ 0, 0, 1]] )
    
    return RZ

# Pack and unpack ROS messages for HumanPose2D and HandPose2D
def packHumanPose2DMsg(now, keypoints):
    # Create ROS msg based on the keypoints
    msg = HumanPose2D()
    msg.header.stamp = now
    # TODO: How to make this shorter, based on mapping? [Can be moved to utils, basically just requires message translations]
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
    msg.r_ankle.x = keypoints[16][0]; msg.r_ankle.y = keypoints[16][1]
    return msg

def unpackHumanPose2DMsg(msg):
    hpe_pxs = []
    hpe_pxs.append((msg.nose.x, msg.nose.y))
    hpe_pxs.append((msg.l_eye.x, msg.l_eye.y))
    hpe_pxs.append((msg.r_eye.x, msg.r_eye.y))
    hpe_pxs.append((msg.l_ear.x, msg.l_ear.y))
    hpe_pxs.append((msg.r_ear.x, msg.r_ear.y))
    hpe_pxs.append((msg.l_shoulder.x, msg.l_shoulder.y))
    hpe_pxs.append((msg.r_shoulder.x, msg.r_shoulder.y))
    hpe_pxs.append((msg.l_elbow.x, msg.l_elbow.y))
    hpe_pxs.append((msg.r_elbow.x, msg.r_elbow.y))
    hpe_pxs.append((msg.l_wrist.x, msg.l_wrist.y))
    hpe_pxs.append((msg.r_wrist.x, msg.r_wrist.y))
    hpe_pxs.append((msg.l_hip.x, msg.l_hip.y))
    hpe_pxs.append((msg.r_hip.x, msg.r_hip.y))
    hpe_pxs.append((msg.l_knee.x, msg.l_knee.y))
    hpe_pxs.append((msg.r_knee.x, msg.r_knee.y))
    hpe_pxs.append((msg.l_ankle.x, msg.l_ankle.y))
    hpe_pxs.append((msg.r_ankle.x, msg.r_ankle.y))
    return hpe_pxs

def packHandPose2DMsg(now, keypoints):
    # Create ROS msg based on the keypoints
    msg = HandPose2D()
    msg.header.stamp = now
    msg.wrist.x = keypoints[0][0]; msg.wrist.y = keypoints[0][1]
    msg.thumb0.x = keypoints[1][0]; msg.thumb0.y = keypoints[1][1]
    msg.thumb1.x = keypoints[2][0]; msg.thumb1.y = keypoints[2][1]
    msg.thumb2.x = keypoints[3][0]; msg.thumb2.y = keypoints[3][1]
    msg.thumb3.x = keypoints[4][0]; msg.thumb3.y = keypoints[4][1]
    msg.index0.x = keypoints[5][0]; msg.index0.y = keypoints[5][1]
    msg.index1.x = keypoints[6][0]; msg.index1.y = keypoints[6][1]
    msg.index2.x = keypoints[7][0]; msg.index2.y = keypoints[7][1]
    msg.index3.x = keypoints[8][0]; msg.index3.y = keypoints[8][1]
    msg.middle0.x = keypoints[9][0]; msg.middle0.y = keypoints[9][1]
    msg.middle1.x = keypoints[10][0]; msg.middle1.y = keypoints[10][1]
    msg.middle2.x = keypoints[11][0]; msg.middle2.y = keypoints[11][1]
    msg.middle3.x = keypoints[12][0]; msg.middle3.y = keypoints[12][1]
    msg.ring0.x = keypoints[13][0]; msg.ring0.y = keypoints[13][1]
    msg.ring1.x = keypoints[14][0]; msg.ring1.y = keypoints[14][1]
    msg.ring2.x = keypoints[15][0]; msg.ring2.y = keypoints[15][1]
    msg.ring3.x = keypoints[16][0]; msg.ring3.y = keypoints[16][1]
    msg.pinky0.x = keypoints[17][0]; msg.pinky0.y = keypoints[17][1]
    msg.pinky1.x = keypoints[18][0]; msg.pinky1.y = keypoints[18][1]
    msg.pinky2.x = keypoints[19][0]; msg.pinky2.y = keypoints[19][1]
    msg.pinky3.x = keypoints[20][0]; msg.pinky3.y = keypoints[20][1]
    return msg

def unpackHandPose2DMsg(msg):
    hand_keypoints = []
    hand_keypoints.append((msg.wrist.x, msg.wrist.y))
    hand_keypoints.append((msg.thumb0.x, msg.thumb0.y))
    hand_keypoints.append((msg.thumb1.x, msg.thumb1.y))
    hand_keypoints.append((msg.thumb2.x, msg.thumb2.y))
    hand_keypoints.append((msg.thumb3.x, msg.thumb3.y))
    hand_keypoints.append((msg.index0.x, msg.index0.y))
    hand_keypoints.append((msg.index1.x, msg.index1.y))
    hand_keypoints.append((msg.index2.x, msg.index2.y))
    hand_keypoints.append((msg.index3.x, msg.index3.y))
    hand_keypoints.append((msg.middle0.x, msg.middle0.y))
    hand_keypoints.append((msg.middle1.x, msg.middle1.y))
    hand_keypoints.append((msg.middle2.x, msg.middle2.y))
    hand_keypoints.append((msg.middle3.x, msg.middle3.y))
    hand_keypoints.append((msg.ring0.x, msg.ring0.y))
    hand_keypoints.append((msg.ring1.x, msg.ring1.y))
    hand_keypoints.append((msg.ring2.x, msg.ring2.y))
    hand_keypoints.append((msg.ring3.x, msg.ring3.y))
    hand_keypoints.append((msg.pinky0.x, msg.pinky0.y))
    hand_keypoints.append((msg.pinky1.x, msg.pinky1.y))
    hand_keypoints.append((msg.pinky2.x, msg.pinky2.y))
    hand_keypoints.append((msg.pinky3.x, msg.pinky3.y))
    return hand_keypoints

def dict_to_matrix(data_dict):
    """
    Converts a dictionary with keys x, y, z, and their corresponding lists
    into a numpy matrix of shape (3, len(list_per_key)).

    Parameters:
        data_dict (dict): A dictionary with keys 'x', 'y', and 'z'.

    Returns:
        numpy.ndarray: A matrix with x values in the first row,
                       y values in the second row,
                       z values in the third row.
    """
    return np.array([data_dict['x'], data_dict['y'], data_dict['z']])

# Losing precision here [How to quantify lost precision here?]
def resize_preds_on_original_size(preds, img_size):
    resized_preds = []
    for pred in preds: 
        p_w, p_h = pred[0], pred[1]
        # This resize is bad, because it floors and it is not correct I would say
        # Test floor, test ceil and test average of floor and ceil
        floor_w = np.floor(p_w/224 * img_size[0])
        floor_h = np.floor(p_h/224 * img_size[1])
        ceil_w = np.ceil(p_w/224 * img_size[0])
        ceil_h = np.ceil(p_h/224 * img_size[1])
        avg_w = (floor_w + ceil_w) / 2
        avg_h = (floor_h + ceil_h) / 2
        resized_preds.append([int(avg_w), int(avg_h)])
    return resized_preds
