#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 13:28:49 2019

@author: zl
"""

from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras.applications.resnet50 import preprocess_input,ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras import backend as K
from keras.utils import Sequence
from tqdm import tqdm_notebook
import numpy as np
from img_tool import read_for_validation



  