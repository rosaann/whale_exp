#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:33:27 2019

@author: zl
"""

import torch.utils.data as data
import pandas as pd
import cv2
import os
import random 
import numpy as np
from pandas import read_csv
from os.path import isfile
from PIL import Image as pil_image
from tqdm import tqdm
from math import sqrt
from scipy.ndimage import affine_transform
import pickle
from imagehash import phash
import torch

crop_margin  = 0.05
anisotropy = 2.15 # The horizontal compression ratio
rotate = []
img_shape = (384,384,1)
eps=1e-10

def expand_path(p):
    if isfile('../train/' + p): return '../train/' + p
    if isfile('../test/' + p): return '../test/' + p
    return p

def read_raw_image(p, rotate):
    img = pil_image.open(expand_path(p))
    #img = image.load_img(expand_path(p), target_size=(224, 224))
    if p in rotate: img = img.rotate(180)
    return img

def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    Build a transformation matrix with the specified characteristics.
    """
    rotation        = np.deg2rad(rotation)
    shear           = np.deg2rad(shear)
    rotation_matrix = np.array([[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix    = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix    = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix     = np.array([[1.0/height_zoom, 0, 0], [0, 1.0/width_zoom, 0], [0, 0, 1]])
    shift_matrix    = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))

class WhaleDataSet(data.Dataset):
    def __init__(self):
        self.img_shape = img_shape
        self.tagged = dict([(p,w) for _,p,w in read_csv('../train.csv').to_records()])
        self.submit = [p for _,p,_ in read_csv('../sample_submission.csv').to_records()]
        self.join   = list(self.tagged.keys()) + self.submit
        self.p2size = {}
        for p in tqdm(self.join):
            size      = pil_image.open(expand_path(p)).size
            self.p2size[p] = size
            
        self.load_p2h()
        self.load_h2p()
        self.load_h2w()
        self.load_w2h()
        self.load_t2i()
        
# First try to use lapjv Linear Assignment Problem solver as it is much faster.
# At the time I am writing this, kaggle kernel with custom package fail to commit.
# scipy can be used as a fallback, but it is too slow to run this kernel under the time limit
# As a workaround, use scipy with data partitioning.
# Because algorithm is O(n^3), small partitions are much faster, but not what produced the submitted solution
        try:
            from lap import lapjv
            self.segment = False
        except ImportError:
            print('Module lap not found, emulating with much slower scipy.optimize.linear_sum_assignment')
            self.segment = True
            from scipy.optimize import linear_sum_assignment
    def setupScore(self, score, steps=1000, batch_size=32):
        """
        @param score the cost matrix for the picture matching
        @param steps the number of epoch we are planning with this score matrix
        """
        self.score      = -score # Maximizing the score is the same as minimuzing -score.
        self.steps      = steps
        self.batch_size = batch_size
        #同一类的样本，score初始化一个大值
        for ts in self.w2ts.values():
            idxs = [self.t2i[t] for t in ts]
            for i in idxs:
                for j in idxs:
                    #同一类的两个样本，给高分
                    self.score[i,j] = 10000.0 # Set a large value for matching whales -- eliminates this potential pairing
        self.on_epoch_end()
        
    def __getitem__(self, index):
        start = self.batch_size*index
        end   = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size  = end - start
        assert size > 0
        a     = np.zeros((size,) + img_shape)
        b     = np.zeros((size,) + img_shape)
        c     = np.zeros((size,1), dtype=torch.float)
        j     = start//2
        for i in range(0, size, 2):
            a[i,  :,:,:] = self.read_for_training(self.match[j][0])
            b[i,  :,:,:] = self.read_for_training(self.match[j][1])
            c[i,  0    ] = 1 # This is a match
            a[i+1,:,:,:] = self.read_for_training(self.unmatch[j][0])
            b[i+1,:,:,:] = self.read_for_training(self.unmatch[j][1])
            c[i+1,0    ] = 0 # Different whales
            j           += 1
        return [a,b],c
    
    def on_epoch_end(self):
        if self.steps <= 0: return # Skip this on the last epoch.
        self.steps     -= 1
        self.match      = []
        self.unmatch    = []
        if self.segment:
            # Using slow scipy. Make small batches.
            # Because algorithm is O(n^3), small batches are much faster.
            # However, this does not find the real optimum, just an approximation.
            tmp   = []
            batch = 512
            for start in range(0, self.score.shape[0], batch):
                end = min(self.score.shape[0], start + batch)
                _, x = linear_sum_assignment(self.score[start:end, start:end])
                tmp.append(x + start)
            x = np.concatenate(tmp)
        else:
            _,_,x = lapjv(self.score) # Solve the linear assignment problem
        y = np.arange(len(x),dtype=np.int32)

        # Compute a derangement for matching whales
        # self.match :所有同类的两两组合[(ts1, ts2), (ts3, ts4)]
        for ts in self.w2ts.values():
            d = ts.copy()
            while True:
                random.shuffle(d)
                if not np.any(ts == d): break
            for ab in zip(ts,d): self.match.append(ab)

        # Construct unmatched whale pairs from the LAP solution.
        # self.unmatch :不同类样本的ts两两组合列表[(tsu1, tsu2), (tsu3, tsu4)],x是从匈牙利算法得出的基于score矩阵的结果序列
        for i,j in zip(x,y):
            if i == j:
                print(self.score)
                print(x)
                print(y)
                print(i,j)
            assert i != j
            self.unmatch.append((self.train[i],self.train[j]))

        # Force a different choice for an eventual next epoch.
        self.score[x,y] = 10000.0
        self.score[y,x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        # print(len(self.match), len(train), len(self.unmatch), len(train))
        assert len(self.match) == len(self.train) and len(self.unmatch) == len(self.train)
        
        
    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1)//self.batch_size

    def load_p2h(self):
        if isfile('p2h.pickle'):
            with open('p2h.pickle', 'rb') as f:
                self.p2h = pickle.load(f)
        else:
            # Compute phash for each image in the training and test set.
            self.p2h = {}
            for p in tqdm(self.join):
                img    = pil_image.open(expand_path(p))
                h      = phash(img)
                self.p2h[p] = h

            # Find all images associated with a given phash value.
            self.h2ps = {}
            for p,h in self.p2h.items():
                if h not in self.h2ps: self.h2ps[h] = []
                if p not in self.h2ps[h]: self.h2ps[h].append(p)

            # Find all distinct phash values
            hs = list(self.h2ps.keys())

            # If the images are close enough, associate the two phash values (this is the slow part: n^2 algorithm)
            self.h2h = {}
            for i,h1 in enumerate(tqdm(hs)):
                for h2 in hs[:i]:
                    if h1-h2 <= 6 and self.match(h1, h2):
                        s1 = str(h1)
                        s2 = str(h2)
                        if s1 < s2: s1,s2 = s2,s1
                        self.h2h[s1] = s2

            # Group together images with equivalent phash, and replace by string format of phash (faster and more readable)
            for p,h in self.p2h.items():
                h = str(h)
                if h in self.h2h: h = self.h2h[h]
                self.p2h[p] = h
        
            pickle.dump(self.p2h, open('p2h.pickle', "wb")) 
            with open('p2h.pickle', 'wb') as handle:
                pickle.dump(self.p2h, handle)
                
    def load_h2p(self):
        self.h2ps = {}
        for p,h in self.p2h.items():
            if h not in self.h2ps: self.h2ps[h] = []
            if p not in self.h2ps[h]: self.h2ps[h].append(p)   
            
        self.h2p = {}
        for h,ps in self.h2ps.items(): self.h2p[h] = self.prefer(ps)
    def load_w2ts_before_train(self):
        # Map whale id to the list of associated training picture hash value
        self.w2ts = {}
        for w,hs in self.w2hs.items():
          for h in hs:
            if h in self.train_set:
                if w not in self.w2ts: self.w2ts[w] = []
                if h not in self.w2ts[w]: self.w2ts[w].append(h)
        for w,ts in self.w2ts.items(): self.w2ts[w] = np.array(ts)
        
        # Map training picture hash value to index in 'train' array    
        t2i  = {}
        for i,t in enumerate(self.train): t2i[t] = i   
        
    def load_h2w(self):
        self.h2ws = {}
        new_whale = 'new_whale'
        for p,w in self.tagged.items():
            if w != new_whale: # Use only identified whales
                h = self.p2h[p]
                if h not in self.h2ws: self.h2ws[h] = []
                if w not in self.h2ws[h]: self.h2ws[h].append(w)
        for h,ws in self.h2ws.items():
            if len(ws) > 1:
                self.h2ws[h] = sorted(ws)
                
    def load_w2h(self):
        self.w2hs = {}
        for h,ws in self.h2ws.items():
            if len(ws) == 1: # Use only unambiguous pictures
            #   if h2p[h] in exclude:
            #       print(h) # Skip excluded images
            #   else:
                w = ws[0]
                if w not in self.w2hs: self.w2hs[w] = []
                if h not in self.w2hs[w]: self.w2hs[w].append(h)
        for w,hs in self.w2hs.items():
            if len(hs) > 1:
                self.w2hs[w] = sorted(hs)
    def load_known(self):
        # Find elements from training sets not 'new_whale'
        self.h2ws = {}
        for p,w in self.tagged.items():
            if w != 'new_whale': # Use only identified whales
                h = self.p2h[p]
                if h not in self.h2ws: self.h2ws[h] = []
                if w not in self.h2ws[h]: self.h2ws[h].append(w)
        self.known = sorted(list(self.h2ws.keys()))

        # Dictionary of picture indices
        self.h2i   = {}
        for i,h in enumerate(self.known): self.h2i[h] = i           
    def load_t2i(self):
        # Find the list of training images, keep only whales with at least two images.
        self.train = [] # A list of training image ids
        for hs in self.w2hs.values():
            if len(hs) > 1:
                self.train += hs
        random.shuffle(self.train)
        self.train_set = set(self.train)

        self.w2ts = {} # Associate the image ids from train to each whale id.
        for w,hs in self.w2hs.items():
            for h in hs:
                if h in self.train_set:
                    if w not in self.w2ts: self.w2ts[w] = []
                    if h not in self.w2ts[w]: self.w2ts[w].append(h)
        for w,ts in self.w2ts.items(): self.w2ts[w] = np.array(ts)
    
        self.t2i = {} # The position in train of each training image id
        for i,t in enumerate(self.train): self.t2i[t] = i
                
# Two phash values are considered duplicate if, for all associated image pairs:
# 1) They have the same mode and size;
# 2) After normalizing the pixel to zero mean and variance 1.0, the mean square error does not exceed 0.1
    def match(self, h1, h2):
       for p1 in self.h2ps[h1]:
          for p2 in self.h2ps[h2]:
            i1 =  pil_image.open(expand_path(p1))
            i2 =  pil_image.open(expand_path(p2))
            if i1.mode != i2.mode or i1.size != i2.size: return False
            a1 = np.array(i1)
            a1 = a1 - a1.mean()
            a1 = a1/sqrt((a1**2).mean())
            a2 = np.array(i2)
            a2 = a2 - a2.mean()
            a2 = a2/sqrt((a2**2).mean())
            a  = ((a1 - a2)**2).mean()
            if a > 0.1: return False
       return True 
   
    def prefer(self, ps):
        if len(ps) == 1: return ps[0]
        best_p = ps[0]
        best_s = self.p2size[best_p]
        for i in range(1, len(ps)):
            p = ps[i]
            s = self.p2size[p]
            if s[0]*s[1] > best_s[0]*best_s[1]: # Select the image with highest resolution
                best_p = p
                best_s = s
        return best_p
        
    def read_cropped_image(self, p, augment):
        """
        @param p : the name of the picture to read
        @param augment: True/False if data augmentation should be performed
        @return a numpy array with the transformed image
        """
        # If an image id was given, convert to filename
        if p in self.h2p: p = self.h2p[p]
        size_x,size_y = self.p2size[p]
    
            # Determine the region of the original image we want to capture based on the bounding box.
            #   x0,y0,x1,y1   = p2bb[p]  #先不考虑bounding box
        x0,y0,x1,y1 = 0, 0, size_x, size_y
    
        if p in rotate: x0, y0, x1, y1 = size_x - x1, size_y - y1, size_x - x0, size_y - y0
        dx            = x1 - x0
        dy            = y1 - y0
        x0           -= dx*crop_margin
        x1           += dx*crop_margin + 1
        y0           -= dy*crop_margin
        y1           += dy*crop_margin + 1
        if (x0 < 0     ): x0 = 0
        if (x1 > size_x): x1 = size_x
        if (y0 < 0     ): y0 = 0
        if (y1 > size_y): y1 = size_y
        dx            = x1 - x0
        dy            = y1 - y0
        if dx > dy*anisotropy:
            dy  = 0.5*(dx/anisotropy - dy)
            y0 -= dy
            y1 += dy
        else:
            dx  = 0.5*(dy*anisotropy - dx)
            x0 -= dx
            x1 += dx

        # Generate the transformation matrix
        trans = np.array([[1, 0, -0.5*img_shape[0]], [0, 1, -0.5*img_shape[1]], [0, 0, 1]])
        trans = np.dot(np.array([[(y1 - y0)/img_shape[0], 0, 0], [0, (x1 - x0)/img_shape[1], 0], [0, 0, 1]]), trans)
        if augment:
          trans = np.dot(build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.8, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(-0.05*(y1 - y0), 0.05*(y1 - y0)),
            random.uniform(-0.05*(x1 - x0), 0.05*(x1 - x0))
            ), trans)
        trans = np.dot(np.array([[1, 0, 0.5*(y1 + y0)], [0, 1, 0.5*(x1 + x0)], [0, 0, 1]]), trans)

        # Read the image, transform to black and white and comvert to numpy array
        img   = read_raw_image(p, rotate).convert('L')
     #   img   = img_to_array(img)
        img = np.array(img)
       # img = np.expand_dims(img, axis=0)
        #  img = preprocess_input(img)
        # Apply affine transformation
        matrix = trans[:2,:2]
        offset = trans[:2,2]
        print('pre shape ', img.shape, ' out ', img.shape[:-1])
    #    img    = img.reshape(img.shape[:-1])
        img    = affine_transform(img, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant', cval=np.average(img))
        img    = img.reshape(img_shape)

        # Normalize to zero mean and unit variance
        img  -= np.mean(img, keepdims=True)
        img  /= np.std(img, keepdims=True) + eps
        return img

    def read_for_training(self, p):
        """
        Read and preprocess an image with data augmentation (random transform).
        """
        return self.read_cropped_image(self, p, True)

    def read_for_validation(self,p):
        """
        Read and preprocess an image without data augmentation (use for testing).
        """
        return self.read_cropped_image(p, False )    
        
        
# Test on a batch of 32 with random costs.
#score = np.random.random_sample(size=(len(train),len(train)))
#data = TrainingData(score)
#(a, b), c = data[0]
#a.shape, b.shape, c.shape
# A Keras generator to evaluate only the BRANCH MODEL
class FeatureGen(data.Dataset):
    def __init__(self, data,train_dataset, batch_size=64, verbose=1):
        super(FeatureGen, self).__init__()
        self.data       = data
        self.batch_size = batch_size
        self.verbose    = verbose
        self.train_dataset = train_dataset
        
        if self.verbose > 0: self.progress = tqdm(total=len(self), desc='Features')
    def __getitem__(self, index):
        start = self.batch_size*index
        size  = min(len(self.data) - start, self.batch_size)
        a     = np.zeros((size,) + img_shape)
        for i in range(size): a[i,:,:,:] = self.train_dataset.read_for_validation(self.data[start + i])
        if self.verbose > 0: 
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return a
    def __len__(self):
        return (len(self.data) + self.batch_size - 1)//self.batch_size   

# A Keras generator to evaluate on the HEAD MODEL on features already pre-computed.
# It computes only the upper triangular matrix of the cost matrix if y is None.
class ScoreGen(data.Dataset):
    def __init__(self, x, y=None, batch_size=2048, verbose=1):
        super(ScoreGen, self).__init__()
        self.x          = x
        self.y          = y
        self.batch_size = batch_size
        self.verbose    = verbose
        if y is None:
            self.y           = self.x
            #self.ix,是上三角矩阵的x, self.iy是上三角矩阵的y
            self.ix, self.iy = np.triu_indices(x.shape[0],1)
        else:
            self.iy, self.ix = np.indices((y.shape[0],x.shape[0]))
            self.ix          = self.ix.reshape((self.ix.size,))
            self.iy          = self.iy.reshape((self.iy.size,))
        self.subbatch = (len(self.x) + self.batch_size - 1)//self.batch_size
        if self.verbose > 0: self.progress = tqdm(total=len(self), desc='Scores')
    def __getitem__(self, index):
        start = index*self.batch_size
        end   = min(start + self.batch_size, len(self.ix))
        a     = self.y[self.iy[start:end],:]
        b     = self.x[self.ix[start:end],:]
        if self.verbose > 0: 
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        #每回输出一对特征，这一对的序号为上三角矩阵的横纵坐标
        return [a,b]
    def __len__(self):
        return (len(self.ix) + self.batch_size - 1)//self.batch_size





    
        
        
        
        