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

class Sub_Block(nn.Model):
    def __init__(self,layers_in, layers_out):
        super(Sub_Block, self).__init__()
        self.x = nn.BatchNorm2d(layers_in)
        layers = []
        layers += [nn.Conv2d(layers_in, layers_out, kernel_size=1), nn.ReLU(inplace=True)]
        layers += [nn.BatchNorm2d(layers_out)]
        layers += [nn.Conv2d(layers_out, layers_out, kernel_size=3), nn.ReLU(inplace=True)]
        layers += [nn.BatchNorm2d(layers_out)]
        layers += [nn.Conv2d(layers_out, layers_in, kernel_size=1), nn.ReLU(inplace=True)]
        
        self.y = layers
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        out_x = self.x(x)
        out_y = self.y(out_x)
        out = torch.add(out_x, out_y)
        out = self.act(out)
        return out
        
class Branch_Model(nn.Model):
    def __init__(self):
        super(Branch_Model, self).__init__()
        self.layers = []
        self.layers += [nn.Conv2d(1, 64, kernel_size=9, stride=2), nn.ReLU(inplace=True)]
        self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        for _ in range(2):
            self.layers += [nn.BatchNorm2d(64), nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(inplace=True)]
        self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.layers += [nn.BatchNorm2d(64)]
        self.layers += [nn.Conv2d(64, 128, kernel_size=1), nn.ReLU(inplace=True)]
        for _ in range(4):
            self.layers += Sub_Block(128, 64)
            
        self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.layers += [nn.BatchNorm2d(128)]
        self.layers += [nn.Conv2d(128, 256, kernel_size=1), nn.ReLU(inplace=True)]
        for _ in range(4):
            self.layers += Sub_Block(256, 64)    
            
        self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.layers += [nn.BatchNorm2d(256)]
        self.layers += [nn.Conv2d(256, 384, kernel_size=1), nn.ReLU(inplace=True)]
        for _ in range(4):
            self.layers += Sub_Block(384, 96)  
            
        self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.layers += [nn.BatchNorm2d(384)]
        self.layers += [nn.Conv2d(384, 512, kernel_size=1), nn.ReLU(inplace=True)]
        for _ in range(4):
            self.layers += Sub_Block(512, 128)  
            
    def forward(self, x):
        out = self.layers(x)        
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        return out

        

        