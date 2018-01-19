# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:25:30 2018

@author: gamer
"""

import numpy as np
from skimage import transform
from scipy.misc import imshow
import config

#returns preprocessed value of most recent frame
def preprocess(frames):
	return resize(grayscale(np.maximum(frames[0], frames[1]))).reshape(1,-1)

#Takes in an rgb image returns the grayscale
def grayscale(frame):
    R = frame[:,:, 0]
    G = frame[:,:, 1]
    B = frame[:,:, 2]
    return 0.2989*R+0.5870*G + 0.1140*B
    
def get_luminescence(frame):
	R = frame[:,:, 0]
	G = frame[:,:, 1]
	B = frame[:,:, 2]
	return (0.2126*R + 0.7152*G + 0.0722*B).astype(int)


def resize(lum_frame):
	return transform.resize(lum_frame,config.STATE_DIM,mode='reflect')
 
def show(frame):
    imshow(frame)