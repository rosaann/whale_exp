#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 17:26:48 2019

@author: zl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.other_layers.l2norm import L2Norm
import numpy as np
class Branch_Model(nn.Model):
    def __init__(self):
        super(Branch_Model, self).__init__()
        self.layers = []
        self.layers += [nn.Conv2d(3, 64, kernel_size=9, stride=2), nn.ReLU(inplace=True)]
        self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        for _ in range(2):
            self.layers += [nn.BatchNorm2d(64), nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(inplace=True)]
        