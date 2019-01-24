#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:23:42 2019

@author: zl
"""

from keras.preprocessing.image import img_to_array,array_to_img
from scipy.ndimage import affine_transform
from keras.applications.resnet50 import preprocess_input,ResNet50
from keras.preprocessing import image
from os.path import isfile
import numpy as np
import random

# Determise the size of each image
def expand_path(p):
    if isfile('../train/' + p): return '../train/' + p
    if isfile('../test/' + p): return '../test/' + p
    return p


