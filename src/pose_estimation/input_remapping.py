#!/usr/bin/python3

import numpy as np 
import rospy

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