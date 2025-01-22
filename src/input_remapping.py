#!/usr/bin/python3

import numpy as np 
import rospy
from geometry_msgs.msg import Vector3, PoseStamped

# This is just simple input remapping for the keypoints and anchor points
def createOmatrix(nk, nu, ap_ix, mp_ix):
    """
        Create the O matrix for the input remapping.
        Input: 
            nu: number of inputs
            nk: number of keypoints
            ap_ix: indices of anchor points
            mp_ix: indices of mapping points
        Output:
            O: O matrix
    """ 
    O = np.zeros((nk, nu))
    for i, a in enumerate(ap_ix):
        O[a, i] = 1
    for i, m in enumerate(mp_ix):
        O[m, i] = -1
    return O

def createUmatrix(Hp, O): 
    u_hat = np.matmul(Hp, O)
    return u_hat

def tfU2Vect3(u): 
    """
        Transform the columns of the input matrix to Vector3 messages.
        Input: 
            u: input matrix
        Output:
            vects: list of Vector3 messages
    """
    vects = []
    for i in range(u.shape[1]): 
        vect = Vector3()
        vect.x = u[0, i]
        vect.y = u[1, i]
        vect.z = u[2, i]
        vects.append(vect)
    return vects

def tfU2Pose(u): 
    """
        Transform the columns of the input matrix to Pose messages.
        Input: 
            u: input matrix
        Output:
            poses: list of Pose messages
    """
    poses = []
    for i in range(u.shape[1]): 
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "camera_color_optical_frame"
        pose.pose.position.x = u[0, i]
        pose.pose.position.y = u[1, i]
        pose.pose.position.z = u[2, i]
        poses.append(pose)
    return poses
