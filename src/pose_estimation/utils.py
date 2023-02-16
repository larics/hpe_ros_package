#!/usr/bin/python3

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
