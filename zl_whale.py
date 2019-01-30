#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:25:37 2019

@author: zl
"""
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import torch.backends.cudnn as cudnn
from model import Whole_Model, ModelLoss
from whale_dataset import FeatureGen, WhaleDataSet,ScoreGen
from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.optim import lr_scheduler

class Whale(object):
    def __init__(self):
        self.model = Whole_Model()
        self.model.summary()
        self.model.branch_model.summary()
        self.model.header_model.summary()

        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
            cudnn.benchmark = True  
        self.optimizer = optim.Adam(self.model.parameters(), lr=64e-5)
        self.criterion = ModelLoss(self.use_gpu)

        self.writer = SummaryWriter(log_dir='out/')
        
    def compute_score(verbose=1):
        """
        Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
        """
        features = self.model.branch_model.predict_generator(FeatureGen(train, img_shape,verbose=verbose), max_queue_size=12, workers=6, verbose=0)
        score    = self.model.head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=6, verbose=0)
        score    = score_reshape(score, features)
        return features, score












        