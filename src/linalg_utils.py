import numpy as np

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

def create_homogenous_vector(vector):
    return np.append(vector, 1)

def create_homogenous_matrix(R, t):
    T_ = np.matrix(np.hstack((R, t.reshape(3, 1))))
    T = np.matrix(np.vstack((T_, np.array([0, 0, 0, 1])))).round(5)
    return T


def arrayToVect(array, vect): 
    vect.x = array[0]
    vect.y = array[1]
    vect.z = array[2]
    return vect

def arrayToPoint(array, point):
    point.x = array[0]
    point.y = array[1]
    point.z = array[2]
    return point

def pointToArray(msg): 
    return np.array([msg.x, msg.y, msg.z])