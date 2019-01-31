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
#from models.other_layers.l2norm import L2Norm
import numpy as np

class Sub_Block(nn.Module):
    def __init__(self,layers_in, layers_out):
        super(Sub_Block, self).__init__()
        self.x = nn.BatchNorm2d(layers_in)
        layers = []
        layers += [nn.Conv2d(layers_in, layers_out, kernel_size=1), nn.ReLU(inplace=True)]
        layers += [nn.BatchNorm2d(layers_out)]
        layers += [nn.Conv2d(layers_out, layers_out, kernel_size=3), nn.ReLU(inplace=True)]
        layers += [nn.BatchNorm2d(layers_out)]
        layers += [nn.Conv2d(layers_out, layers_in, kernel_size=1), nn.ReLU(inplace=True)]
        
        self.y = nn.ModuleList(layers)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        out_x = self.x(x)
        out_y = self.y(out_x)
        out = torch.add(out_x, out_y)
        out = self.act(out)
        return out
        
class Branch_Model(nn.Module):
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
            self.layers += [Sub_Block(128, 64)]
            
        self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.layers += [nn.BatchNorm2d(128)]
        self.layers += [nn.Conv2d(128, 256, kernel_size=1), nn.ReLU(inplace=True)]
        for _ in range(4):
            self.layers += [Sub_Block(256, 64)]    
            
        self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.layers += [nn.BatchNorm2d(256)]
        self.layers += [nn.Conv2d(256, 384, kernel_size=1), nn.ReLU(inplace=True)]
        for _ in range(4):
            self.layers += [Sub_Block(384, 96)]  
            
        self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.layers += [nn.BatchNorm2d(384)]
        self.layers += [nn.Conv2d(384, 512, kernel_size=1), nn.ReLU(inplace=True)]
        for _ in range(4):
            self.layers += [Sub_Block(512, 128)  ]
        
        self.layers = nn.ModuleList(self.layers)
    def forward(self, x):
        out = self.layers(x)        
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        return out
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class Header_Model(nn.Module):
    def __init__(self):
        super(Header_Model, self).__init__()  
        self.layer_1 = nn.ModuleList( [nn.Conv2d(4, 32, kernel_size=(4, 1)), nn.ReLU(inplace=True)])
        self.layer_2 = nn.ModuleList( [nn.Conv2d(32, 32, kernel_size=(1, 32)), nn.Linear(32, 1)])
        self.flatten = Flatten()
        
        self.dense = nn.ModuleList([nn.Linear(1, 1, bias = True), nn.Sigmoid()])

    def forward(self, x):
        x1 = x[0] * x[1]
        x2 = x[0] + x[1]
        x3 = torch.abs(x[0] - x[1])
        x4 = torch.mul(x3, x3)
        out = torch.cat((x1, x2, x3, x4), 0)
        out = out.view(4, -1, 1)
        
        out = self.layer_1(out)
        out = out.view(-1, 32, 1)
        out = self.layer_2(out)
        out = self.flatten(out)

        out = self.dense(out)
        return out
    
class Whole_Model(nn.Module):
    def __init__(self):
        super(Whole_Model, self).__init__()  
        self.branch_model = Branch_Model()
        self.header_model = Header_Model()
    
    def forward(self, x):
        xa = self.branch_model(x[0])
        xb = self.branch_model(x[1])
        x = self.header_model([xa, xb])
        return x
    
    
class ModelLoss(nn.Module):
    def __init__(self, use_gpu=True):
        super(ModelLoss, self).__init__()
        self.use_gpu = use_gpu
        self.cri = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        conf_data = predictions

        
        conf_t = torch.from_numpy( targets).type(torch.cuda.LongTensor)
        #conf_t = torch.from_numpy( targets)
        if self.use_gpu:
            conf_t = conf_t.cuda()
            conf_data = conf_data.cuda()

        loss_c =self.cri(conf_data, conf_t)
     #   print('loss_c ', loss_c)
        return loss_c
    
    