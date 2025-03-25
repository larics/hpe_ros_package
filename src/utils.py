#!/usr/bin/python3

from hpe_ros_msgs.msg import HumanPose2D, HandPose2D, HumanPose3D, HandPose3D, TorsoJointPositions
from geometry_msgs.msg import Vector3, Point, Twist, Transform
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np

def limitCmd(cmd, upperLimit, lowerLimit):
    if cmd > upperLimit: 
        cmd = upperLimit
    if cmd < lowerLimit: 
        cmd = lowerLimit
    return cmd

# ROS msgs packers and unpackers
def packTorsoPositionMsg(now, keypoints):
    msg = TorsoJointPositions()
    msg.header.stamp = now
    msg.frame_id.data = "world"
    msg.thorax = arrayToVect([keypoints[0, 1], keypoints[1, 1], keypoints[2, 5]], msg.thorax)
    msg.left_shoulder = arrayToVect(keypoints[:, 5], msg.left_shoulder)
    msg.right_shoulder = arrayToVect(keypoints[:, 6], msg.right_shoulder)
    msg.right_elbow = arrayToVect(keypoints[:, 7], msg.right_elbow)
    msg.left_elbow = arrayToVect(keypoints[:, 8], msg.left_elbow)
    msg.right_wrist = arrayToVect(keypoints[:, 10], msg.right_wrist)
    msg.left_wrist = arrayToVect(keypoints[:, 9], msg.left_wrist)
    #msg.success = True
    return msg

def packSimpleTorso3DMsg(now, bD):
    msg = TorsoJointPositions()
    msg.header = now
    msg.frame_id.data = "camera_color_frame"
    msg.thorax = Vector3(bD.T[0, 0], bD.T[1,0], bD.T[2,0])
    msg.left_shoulder = Vector3(bD.T[0, 1], bD.T[1,1], bD.T[2,1])
    msg.right_shoulder = Vector3(bD.T[0, 2], bD.T[1,2], bD.T[2,2])
    msg.left_elbow = Vector3(bD.T[0, 3], bD.T[1,3], bD.T[2,3])
    msg.right_elbow = Vector3(bD.T[0, 4], bD.T[1,4], bD.T[2,4])
    msg.left_wrist = Vector3(bD.T[0, 5], bD.T[1,5], bD.T[2,5])
    msg.right_wrist = Vector3(bD.T[0, 6], bD.T[1,6], bD.T[2,6])
    msg.success.data = True
    return msg

def packTorso3DMsg(self, pos_named, header): 
    # TODO: Pack this into packTorsoPoseMsg method
    msg = TorsoJointPositions()
    msg.header          = header
    msg.frame_id.data   = "camera_color_frame"
    try:
        # COCO doesn't have THORAX! TODO: modify this to fit coco and body25 
        if self.coco or self.body25: 
            thorax = Vector3((pos_named["l_shoulder"][0] + pos_named["r_shoulder"][0])/2, 
                             (pos_named["l_shoulder"][1] + pos_named["r_shoulder"][1])/2, 
                             (pos_named["l_shoulder"][2] + pos_named["r_shoulder"][2])/2)
            msg.thorax = thorax
        else: 
            msg.thorax      = Vector3(pos_named["thorax"][0], pos_named["thorax"][1], pos_named["thorax"][2])
        msg.left_elbow      = Vector3(pos_named["l_elbow"][0], pos_named["l_elbow"][1], pos_named["l_elbow"][2])
        msg.right_elbow     = Vector3(pos_named["r_elbow"][0], pos_named["r_elbow"][1], pos_named["r_elbow"][2])
        msg.left_shoulder   = Vector3(pos_named["l_shoulder"][0], pos_named["l_shoulder"][1], pos_named["l_shoulder"][2])
        msg.right_shoulder  = Vector3(pos_named["r_shoulder"][0], pos_named["r_shoulder"][1], pos_named["r_shoulder"][2])
        msg.left_wrist      = Vector3(pos_named["l_wrist"][0], pos_named["l_wrist"][1], pos_named["l_wrist"][2])
        msg.right_wrist     = Vector3(pos_named["r_wrist"][0], pos_named["r_wrist"][1], pos_named["r_wrist"][2])
        msg.success.data = True
    except Exception as e:
        msg.success.data = False 

    return msg

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

def packHumanPose3DMsg(now, keypoints):
    msg = HumanPose3D()
    msg.header.stamp = now
    msg.nose        = arrayToPoint(keypoints[:, 0], msg.nose)
    msg.l_eye       = arrayToPoint(keypoints[:, 1], msg.l_eye)
    msg.r_eye       = arrayToPoint(keypoints[:, 2], msg.r_eye)
    msg.l_ear       = arrayToPoint(keypoints[:, 3], msg.l_ear)
    msg.r_ear       = arrayToPoint(keypoints[:, 4], msg.r_ear)
    msg.l_shoulder  = arrayToPoint(keypoints[:, 5], msg.l_shoulder)
    msg.r_shoulder  = arrayToPoint(keypoints[:, 6], msg.r_shoulder)
    msg.l_elbow     = arrayToPoint(keypoints[:, 7], msg.l_elbow)
    msg.r_elbow     = arrayToPoint(keypoints[:, 8], msg.r_elbow)
    msg.l_wrist     = arrayToPoint(keypoints[:, 9], msg.l_wrist)
    msg.r_wrist     = arrayToPoint(keypoints[:, 10], msg.r_wrist)
    msg.l_hip       = arrayToPoint(keypoints[:, 11], msg.l_hip)
    msg.r_hip       = arrayToPoint(keypoints[:, 12], msg.r_hip)
    msg.l_knee      = arrayToPoint(keypoints[:, 13], msg.l_knee)
    msg.r_knee      = arrayToPoint(keypoints[:, 14], msg.r_knee)
    msg.l_ankle     = arrayToPoint(keypoints[:, 15], msg.l_ankle)
    msg.r_ankle     = arrayToPoint(keypoints[:, 16], msg.r_ankle)
    return msg

def packOPHumanPose3DMsg(now, keypoints):
    msg = HumanPose3D()
    msg.header.stamp = now
    msg.nose = dictValByKeyToPoint(keypoints, 'nose', msg.nose)
    msg.neck = dictValByKeyToPoint(keypoints, 'neck', msg.neck)
    msg.r_eye = dictValByKeyToPoint(keypoints, 'r_eye', msg.r_eye)
    msg.l_eye = dictValByKeyToPoint(keypoints, 'l_eye', msg.l_eye)
    msg.r_ear = dictValByKeyToPoint(keypoints, 'r_ear', msg.r_ear)
    msg.l_ear = dictValByKeyToPoint(keypoints, 'l_ear', msg.l_ear)
    msg.r_shoulder = dictValByKeyToPoint(keypoints, 'r_shoulder', msg.r_shoulder)
    msg.l_shoulder = dictValByKeyToPoint(keypoints, 'l_shoulder', msg.l_shoulder)
    msg.r_elbow = dictValByKeyToPoint(keypoints, 'r_elbow', msg.r_elbow)
    msg.l_elbow = dictValByKeyToPoint(keypoints, 'l_elbow', msg.l_elbow)
    msg.r_wrist = dictValByKeyToPoint(keypoints, 'r_wrist', msg.r_wrist)
    msg.l_wrist = dictValByKeyToPoint(keypoints, 'l_wrist', msg.l_wrist)
    msg.r_hip = dictValByKeyToPoint(keypoints, 'r_hip', msg.r_hip)
    msg.l_hip = dictValByKeyToPoint(keypoints, 'l_hip', msg.l_hip)
    msg.r_knee = dictValByKeyToPoint(keypoints, 'r_knee', msg.r_knee)
    msg.l_knee = dictValByKeyToPoint(keypoints, 'l_knee', msg.l_knee)
    msg.r_ankle = dictValByKeyToPoint(keypoints, 'r_ankle', msg.r_ankle)
    msg.l_ankle = dictValByKeyToPoint(keypoints, 'l_ankle', msg.l_ankle)
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

def packHandPose3DMsg(now, keypoints):
    msg = HandPose3D()
    msg.header.stamp = now
    msg.wrist = dictValByKeyToPoint(keypoints, 'wrist', msg.wrist)
    msg.thumb0 = dictValByKeyToPoint(keypoints, 'thumb0', msg.thumb0)
    msg.thumb1 = dictValByKeyToPoint(keypoints, 'thumb1', msg.thumb1)
    msg.thumb2 = dictValByKeyToPoint(keypoints, 'thumb2', msg.thumb2)
    msg.thumb3 = dictValByKeyToPoint(keypoints, 'thumb3', msg.thumb3)
    msg.index0 = dictValByKeyToPoint(keypoints, 'index0', msg.index0)
    msg.index1 = dictValByKeyToPoint(keypoints, 'index1', msg.index1)
    msg.index2 = dictValByKeyToPoint(keypoints, 'index2', msg.index2)
    msg.index3 = dictValByKeyToPoint(keypoints, 'index3', msg.index3)
    msg.middle0 = dictValByKeyToPoint(keypoints, 'middle0', msg.middle0)
    msg.middle1 = dictValByKeyToPoint(keypoints, 'middle1', msg.middle1)
    msg.middle2 = dictValByKeyToPoint(keypoints, 'middle2', msg.middle2)
    msg.middle3 = dictValByKeyToPoint(keypoints, 'middle3', msg.middle3)
    msg.ring0 = dictValByKeyToPoint(keypoints, 'ring0', msg.ring0)
    msg.ring1 = dictValByKeyToPoint(keypoints, 'ring1', msg.ring1)
    msg.ring2 = dictValByKeyToPoint(keypoints, 'ring2', msg.ring2)
    msg.ring3 = dictValByKeyToPoint(keypoints, 'ring3', msg.ring3)
    msg.pinky0 = dictValByKeyToPoint(keypoints, 'pinky0', msg.pinky0)
    msg.pinky1 = dictValByKeyToPoint(keypoints, 'pinky1', msg.pinky1)
    msg.pinky2 = dictValByKeyToPoint(keypoints, 'pinky2', msg.pinky2)
    msg.pinky3 = dictValByKeyToPoint(keypoints, 'pinky3', msg.pinky3)
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

def dictValByKeyToPoint(dict, key, point):
    point.x = dict[key][0]
    point.y = dict[key][1]
    point.z = dict[key][2]
    return point

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

def get_key_by_value(d, value):
    """
    Finds and returns the key corresponding to a given value in a dictionary.

    Parameters:
        d (dict): The dictionary to search.
        value: The value to find the corresponding key for.

    Returns:
        The key associated with the value if found, or None if the value is not in the dictionary.
    """
    for key, val in d.items():
        if val == value:
            return key
    return None

def get_allocation_matrix(n, m): 
    """
    Create an allocation matrix of size n x m. 
    """
    return np.zeros((n, m))

def index_and_operation(A, keys, indices, operation): 
    """ 
    Method prototype, probably should include some overloading. 
    """
    A = get_allocation_matrix((len(keys), 3))

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

def remove_nans(matrix): 
    """
    Swap nans with zeros in a matrix.
    """
    #swap nans with zeros
    matrix[np.isnan(matrix)] = 0
    return matrix

def getZeroTwist(): 
    tw = Twist()
    tw.linear.x = 0
    tw.linear.y = 0
    tw.linear.z = 0
    tw.angular.x = 0
    tw.angular.y = 0
    tw.angular.z = 0
    return tw

def getZeroTransform(): 
    t = Transform()
    t.translation.x = 0
    t.translation.y = 0
    t.translation.z = 0
    t.rotation.x = 0
    t.rotation.y = 0
    t.rotation.z = 0
    t.rotation.w = 1
    return t

# Publish marker array
def publishMarkerArray(self, bD):
    mA = MarkerArray()
    i = 0
    names = ["ls", "rs", "le", "re", "lw", "rw"]
    for v in bD:
        m_ = createMarker(v, i)
        i+=1 
        mA.markers.append(m_)
    return mA

def createMarker(now, v, i):
    m_ = Marker()
    m_.header.frame_id = "camera_color_frame"
    m_.header.stamp = now()
    m_.type = m_.SPHERE
    m_.id = i
    m_.action = m_.ADD
    m_.scale.x = 0.1
    m_.scale.y = 0.1
    m_.scale.z = 0.1
    m_.color.a = 1.0
    m_.color.r = 0.0
    m_.color.g = 1.0
    m_.color.b = 0.0
    m_.pose.position.x = v[0]
    m_.pose.position.y = v[1]
    m_.pose.position.z = v[2]
    m_.pose.orientation.x = 0
    m_.pose.orientation.y = 0
    m_.pose.orientation.z = 0
    m_.pose.orientation.w = 1
    return m_

# TODO: Make this better 4 sure
def create_marker_wth_starting_pt(now, shape, start_p, dist_x, dist_y, dist_z): 
    marker = Marker()
    marker.header.frame_id = "n_thorax"
    marker.header.stamp = now
    marker.ns = "arrow"
    marker.id = 0
    marker.type = shape
    marker.action = Marker.ADD
    marker.pose.position.x = start_p.x
    marker.pose.position.y = start_p.y
    marker.pose.position.z = start_p.z
    # How to transform x,y,z values to the orientation 
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = dist_x
    marker.scale.y = dist_y
    marker.scale.z = dist_z
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    return marker

def createMarkerArrow(stamp, start_point, end_point, i, color=(255, 0, 0)):
    m_ = Marker()
    m_.header.frame_id = "camera_color_frame"
    m_.header.stamp = stamp
    m_.type = m_.ARROW
    m_.id = i
    m_.action = m_.ADD
    m_.scale.x = 0.02  # shaft diameter
    m_.scale.y = 0.1  # head diameter
    m_.scale.z = 0.1  # head length
    m_.color.a = 1.0
    m_.color.r = color[0]
    m_.color.g = color[1]
    m_.color.b = color[2]
    pt1 = Point()
    pt2 = Point()
    pt1.x = start_point[0]
    pt1.y = start_point[1]
    pt1.z = start_point[2]
    pt2.x = end_point[0]
    pt2.y = end_point[1]
    pt2.z = end_point[2]
    m_.points.append(pt1)
    m_.points.append(pt2)
    return m_