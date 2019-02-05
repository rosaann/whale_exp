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
    def __init__(self,layers_in, layers_out, stride=1):
        super(Sub_Block, self).__init__()
        self.x = nn.BatchNorm2d(layers_in)
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(layers_in, layers_out, kernel_size=1, stride = stride, bias=False))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(layers_out))
        layers.append(nn.Conv2d(layers_out, layers_out, kernel_size=3, stride = stride,padding=1, bias=False))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(layers_out))
        layers.append(nn.Conv2d(layers_out, layers_in, kernel_size=1, stride = stride, bias=False))
        layers.append(nn.ReLU())
        
        self.y = layers
        self.act = nn.ReLU()
    def forward(self, x, phase='train'):
       # print('sub in ', x.shape)
        if phase == 'eval':
            vis_list = []
        out_x = self.x(x)
        if phase == 'eval':
            vis_list.append(out_x)
        out_y = out_x
        for i in range(len(self.y)):
            out_y = (self.y[i])(out_y)
            if phase == 'eval':
                vis_list.append(out_y)
        #    print('out_y ', out_y.shape)
    #    print('out_x ', out_x.shape)

       # out = torch.add(out_x, out_y)
        out_y += out_x
        if phase == 'eval':
                vis_list.append(out_y)
        out_y = self.act(out_y)
        if phase == 'eval':
                vis_list.append(out_y)
     #   print('sub out ', out_y.shape)
        if phase == 'eval':
                return out_y, vis_list
        return out_y
        
class Branch_Model(nn.Module):
    def __init__(self):
        super(Branch_Model, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(1, 64, kernel_size=9, stride=2))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        for _ in range(2):
            self.layers.append(nn.BatchNorm2d(64))
            self.layers.append(nn.Conv2d(64, 64, kernel_size=3))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(nn.Conv2d(64, 128, kernel_size=1))
        self.layers.append(nn.ReLU())
        for _ in range(4):
            self.layers.append(Sub_Block(128, 64))
            
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.BatchNorm2d(128))
        self.layers.append(nn.Conv2d(128, 256, kernel_size=1))
        self.layers.append(nn.ReLU())
        for _ in range(4):
            self.layers.append(Sub_Block(256, 64))    
            
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.BatchNorm2d(256))
        self.layers.append(nn.Conv2d(256, 384, kernel_size=1))
        self.layers.append(nn.ReLU())
        for _ in range(4):
            self.layers.append(Sub_Block(384, 96))  
            
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.BatchNorm2d(384))
        self.layers.append(nn.Conv2d(384, 512, kernel_size=1))
        self.layers.append(nn.ReLU())
        for _ in range(4):
            self.layers.append(Sub_Block(512, 128)  )
             
    def forward(self, x, phase='train'):
       # print('x ', x.shape)
        out = x
        
        if phase == 'eval':
            vis_list = []
        for i in range(len(self.layers)):
            out = (self.layers[i])(out)
            if phase == 'eval':
                vis_list.append(out)
       # print('x2 ', out.shape)
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        if phase == 'eval':
            vis_list.append(out)
            return out, vis_list
        return out
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class Header_Model(nn.Module):
    def __init__(self):
        super(Header_Model, self).__init__()  
        self.layer_1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=(4, 1), padding=0), nn.ReLU())
        #self.layer_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=(1, 32)), nn.Linear(32, 1))
        self.layer_2 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=(32, 1), padding=0))

        self.flatten = Flatten()
        
        self.dense = nn.Sequential(nn.Linear(478, 1, bias = True), nn.Sigmoid())#

    def forward(self, x, phase='train'):
        
       # print('x in head', x.shape)
        x_0 = x[:,0]
        x_1 = x[:,1]
     #   print('x_0 ', x_0.shape)
        x1 = x_0 * x_1
     #   print('x1 ', x1.shape)
        x2 = x_0 + x_1
     #   print('x2 ', x2.shape)
        x3 = torch.abs(x_0 - x_1)
     #   print('x3 ', x3.shape)
        x4 = torch.mul(x3, x3)
      #  print('x4 ', x4.shape)
        out = torch.cat((x1, x2, x3, x4), -1)
     #   print('out1 ', out.shape)
        out = out.view(x.shape[0], 4, -1, 1)
        if phase == 'eval':
            vis_list = []
      #  print('out shape ', out.shape)
        out = self.layer_1(out)
        if phase == 'eval':
            vis_list.append(out)
      #  print('out1 shape ', out.shape)
        out = out.view(x.shape[0],32, -1, 1)
        if phase == 'eval':
            vis_list.append(out)
      #  print('out2 shape ', out.shape)
        out = self.layer_2(out)
        if phase == 'eval':
            vis_list.append(out)
     #   print('out3 shape ', out.shape)
        out = self.flatten(out)
      #  print('out4 shape ', out.shape)
        out = self.dense(out)
     #   print('out5 shape ', out.shape)
        if phase == 'eval':
            return out, vis_list
        return out
    
class Whole_Model(nn.Module):
    def __init__(self):
        super(Whole_Model, self).__init__()  
        self.branch_model = Branch_Model()
        self.header_model = Header_Model()
    
    def forward(self, x, phase='train'):
        if phase == 'eval':
            xa, vis_a = self.branch_model(x[:,0], phase = phase)
            xb, vis_b = self.branch_model(x[:,1], phase = phase)
        else:
            xa = self.branch_model(x[:,0])
            xb = self.branch_model(x[:,1])

        xa = xa.unsqueeze(1)
        xb = xb.unsqueeze(1)
        x = torch.cat((xa,xb), 1)
        
        if phase == 'eval':
            x, vis_x = self.header_model(x, phase = phase)
        else:
            x = self.header_model(x)
            
        if phase == 'eval':
            vis = []
            vis.extend(vis_a)
            vis.extend(vis_b)
            vis.extend(vis_x)
            return x, vis
        return x
    
    
class ModelLoss(nn.Module):
    def __init__(self, use_gpu=True):
        super(ModelLoss, self).__init__()
        self.use_gpu = use_gpu
        self.cri = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        conf_data = predictions

        
        conf_t = torch.from_numpy( targets).type(torch.cuda.FloatTensor)
        #conf_t = torch.from_numpy( targets)
        if self.use_gpu:
            conf_t = conf_t.cuda()
            conf_data = conf_data.cuda()

        loss_c =self.cri(conf_data, conf_t)
     #   print('loss_c ', loss_c)
        return loss_c
    
    