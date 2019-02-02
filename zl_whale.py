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
import torch.utils.data as data
import numpy as np
import random
import gzip
from tqdm import tqdm
from torchsummary import summary

class Whale(object):
    def __init__(self):
        self.model = Whole_Model()
       # summary(self.model, input_size=(3, 255, 255))
       # self.model.branch_model.summary()
       # self.model.header_model.summary()

        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
            cudnn.benchmark = True  
            
        self.train_data = WhaleDataSet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=64e-5)
        self.criterion = ModelLoss(self.use_gpu)

        self.writer = SummaryWriter(log_dir='out/')
    
    def score_reshape(self, score, x, y=None):
        """
        Tranformed the packed matrix 'score' into a square matrix.
        @param score the packed matrix
        @param x the first image feature tensor
        @param y the second image feature tensor if different from x
        @result the square matrix
        """
        if y is None:
            # When y is None, score is a packed upper triangular matrix.
            # Unpack, and transpose to form the symmetrical lower triangular matrix.
            m = np.zeros((x.shape[0],x.shape[0]), dtype=torch.float)
            m[np.triu_indices(x.shape[0],1)] = score.squeeze()
            m += m.transpose()
        else:
            m        = np.zeros((y.shape[0],x.shape[0]), dtype=torch.float)
            iy,ix    = np.indices((y.shape[0],x.shape[0]))
            ix       = ix.reshape((ix.size,))
            iy       = iy.reshape((iy.size,))
            m[iy,ix] = score.squeeze()
        return m    
    def compute_score(self, verbose=1):
        """
        Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
        """
        
        feature_data = FeatureGen(self.train_data.train, self.train_data,verbose=verbose)
        feature_dataset = data.DataLoader(feature_data, 32, num_workers= 8,
                        shuffle=False, pin_memory=False)
        
        features = []
        i = 0
        self.model.branch_model.eval()
        for images in feature_dataset:
            if self.use_gpu:
                images = Variable(images.cuda().float())
            f =self.model.branch_model(images)
            print('f ', f.shape, ' ', i)
            i += 1
            features.extend( f.cpu().data )     
                
        score_data = ScoreGen(features, verbose=verbose)
        score_dataset = data.DataLoader(score_data, 32, num_workers= 8,
                        shuffle=False, pin_memory=False)
        
        score = []
        for t_features in score_dataset:
            score.extend(self.model.head_model(t_features))

        score = self.score_reshape(score, features)
        
        #features = self.model.branch_model.predict_generator(FeatureGen(self.dataset.train, self.dataset,verbose=verbose), max_queue_size=12, workers=6, verbose=0)
       # score    = self.model.head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=6, verbose=0)
       # score    = score_reshape(score, features)
        return features, score


    def make_steps(self, step, ampl):
        """
        Perform training epochs
        @param step Number of epochs to perform
        @param ampl the K, the randomized component of the score matrix.
        """
        global w2ts, t2i, steps, features, score, histories
    
        # shuffle the training pictures
        #train 是w--hs list,过滤了只有一个样本的w
        random.shuffle(self.train_data.train)
    
        self.train_data.load_w2ts_before_train()

     
        # Compute the match score for each picture pair
        features, score = self.compute_score()
        self.train_data.setupScore(score + ampl*np.random.random_sample(size=score.shape), steps=step, batch_size=30)
        train_dataset = data.DataLoader(self.train_data, 30, num_workers= 8,
                        shuffle=False, pin_memory=True)
        for epoch in step:
            for image_pairs, ts in train_dataset:
                if self.use_gpu:
                    image_pairs = Variable(image_pairs.cuda())
                self.model.train()
                out = self.model(image_pairs)
                self.optimizer.zero_grad()
                loss_c = self.criterion(out, ts)
                loss_c.backward()
                self.optimizer.step()
            self.train_data.on_epoch_end()
            train_dataset = data.DataLoader(self.train_data, 30, num_workers= 8,
                        shuffle=False, pin_memory=True)
   
        steps += step
    
    
    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    def train(self):
        file_name = 'mpiotte_model_torch.model'
        
        if False:
            tmp = keras.models.load_model('mpiotte-standard.model')
            model.set_weights(tmp.get_weights())
        else:
            # epoch -> 10
            self.make_steps(10, 1000)
            ampl = 100.0
            for _ in range(10):
                print('noise ampl.  = ', ampl)
                self.make_steps(5, ampl)
                ampl = max(1.0, 100**-0.1*ampl)
                # epoch -> 150
            for _ in range(18): self.make_steps(5, 1.0)
            # epoch -> 200
            self.set_lr( 16e-5)
            for _ in range(10): self.make_steps(5, 0.5)
            # epoch -> 240
            self.set_lr( 4e-5)
            for _ in range(8): self.make_steps(5, 0.25)
            # epoch -> 250
            self.set_lr( 1e-5)
            for _ in range(2): self.make_steps(5, 0.25)
            # epoch -> 300
           # weights = model.get_weights()
           # model, branch_model, head_model = build_model(64e-5,0.0002)
           # model.set_weights(weights)
            for _ in range(10): self.make_steps(5, 1.0)
            # epoch -> 350
            self.set_lr( 16e-5)
            for _ in range(10): self.make_steps(5, 0.5)    
            # epoch -> 390
            self.set_lr( 4e-5)
            for _ in range(8): self.make_steps(5, 0.25)
            # epoch -> 400
            self.set_lr( 1e-5)
            for _ in range(2): self.make_steps(5, 0.25)

            torch.save(self.model.state_dict(), file_name)


    def prepare_submission(self, threshold, filename):
        """
        Generate a Kaggle submission file.
        @param threshold the score given to 'new_whale'
        @param filename the submission file name
        """
        vtop  = 0
        vhigh = 0
        pos   = [0,0,0,0,0,0]
        with gzip.open(filename, 'wt', newline='\n') as f:
            f.write('Image,Id\n')
            for i,p in enumerate(tqdm(self.train_data.submit)):
                t = []
                s = set()
                a = score[i,:]
                for j in list(reversed(np.argsort(a))):
                    h = self.train_data.known[j]
                    if a[j] < threshold and 'new_whale' not in s:
                        pos[len(t)] += 1
                        s.add('new_whale')
                        t.append('new_whale')
                        if len(t) == 5: break;
                    for w in self.train_data.h2ws[h]:
                        assert w != 'new_whale'
                        if w not in s:
                            if a[j] > 1.0:
                                vtop += 1
                            elif a[j] >= threshold:
                                vhigh += 1
                            s.add(w)
                            t.append(w)
                            if len(t) == 5: break;
                    if len(t) == 5: break;
                if 'new_whale' not in s: pos[5] += 1
                assert len(t) == 5 and len(s) == 5
                f.write(p + ',' + ' '.join(t[:5]) + '\n')
        return vtop,vhigh,pos

    def test(self):
        self.train_data.load_known()
        
        feature_data = FeatureGen(self.train_data.known, self.train_data)
        feature_dataset = data.DataLoader(feature_data, 32, num_workers= 8,
                        shuffle=False, pin_memory=True)
        fknown = []
        for images in feature_dataset:
            if self.use_gpu:
                images = Variable(images.cuda())
            fknown += self.model.branch_model(images)   
        ##
        
        feature_data_s = FeatureGen(self.train_data.submit, self.train_data)
        feature_data_s_set = data.DataLoader(feature_data_s, 32, num_workers= 8,
                        shuffle=False, pin_memory=True)      
        fsubmit = []
        for images in feature_data_s_set:
            if self.use_gpu:
                images = Variable(images.cuda())
            fsubmit += self.model.branch_model(images)

        ##
        
        score_data = ScoreGen(fknown, fsubmit)
        score_dataset = data.DataLoader(score_data, 32, num_workers= 8,
                        shuffle=False, pin_memory=True)
        
        score = []
        for t_features in score_dataset:
            score += self.model.head_model(t_features)

        score = self.score_reshape(score, features)

    # Evaluate the model.
    
    # Generate the subsmission file.
        self.prepare_submission(0.99, 'zl_mpiotte-standard_lstm.csv.gz')
        
        
        
        
        
        
        
        
        
        