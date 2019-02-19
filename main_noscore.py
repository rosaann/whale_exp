#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:33:55 2019

@author: zl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 10:02:44 2019

@author: zl
"""

from pandas import read_csv

from PIL import Image as pil_image
#from  import tqdm_notebook
from tqdm import tqdm
import pickle
import numpy as np
from imagehash import phash
from math import sqrt
from keras import backend as K
from keras.preprocessing.image import img_to_array,array_to_img
from scipy.ndimage import affine_transform
from keras.applications.resnet50 import preprocess_input,ResNet50
from keras.preprocessing import image
from keras.utils import plot_model
import random
from os.path import isfile
from img_tool import expand_path
from keras.utils import Sequence
from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model

import keras
#from tqdm import tqdm_notebook
from tensorboardX import SummaryWriter
import torch

tagged = dict([(p,w) for _,p,w in read_csv('../train.csv').to_records()])
submit = [p for _,p,_ in read_csv('../sample_submission.csv').to_records()]
join   = list(tagged.keys()) + submit
#len(tagged),len(submit),len(join),list(tagged.items())[:5],submit[:5]
crop_margin  = 0.05
anisotropy   = 2.15 # The horizontal compression ratio

writer = SummaryWriter(log_dir='out/')

p2size = {}
for p in tqdm(join):
    size      = pil_image.open(expand_path(p)).size
    p2size[p] = size
#len(p2size), list(p2size.items())[:5]
    
# Two phash values are considered duplicate if, for all associated image pairs:
# 1) They have the same mode and size;
# 2) After normalizing the pixel to zero mean and variance 1.0, the mean square error does not exceed 0.1
def match(h1,h2):
    for p1 in h2ps[h1]:
        for p2 in h2ps[h2]:
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

# For each images id, select the prefered image
def prefer(ps):
    if len(ps) == 1: return ps[0]
    best_p = ps[0]
    best_s = p2size[best_p]
    for i in range(1, len(ps)):
        p = ps[i]
        s = p2size[p]
        if s[0]*s[1] > best_s[0]*best_s[1]: # Select the image with highest resolution
            best_p = p
            best_s = s
    return best_p

def read_raw_image(p, rotate):
    img = pil_image.open(expand_path(p))
    #img = image.load_img(expand_path(p), target_size=(224, 224))
    if p in rotate: img = img.rotate(180)
    return img


def read_cropped_image(p, augment):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @return a numpy array with the transformed image
    """
    # If an image id was given, convert to filename
    if p in h2p: p = h2p[p]
    size_x,size_y = p2size[p]
    
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
    img   = img_to_array(img)
  #  img = np.expand_dims(img, axis=0)
  #  img = preprocess_input(img)
    # Apply affine transformation
    matrix = trans[:2,:2]
    offset = trans[:2,2]
    img    = img.reshape(img.shape[:-1])
    img    = affine_transform(img, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant', cval=np.average(img))
    img    = img.reshape(img_shape)

    # Normalize to zero mean and unit variance
    img  -= np.mean(img, keepdims=True)
    img  /= np.std(img, keepdims=True) + K.epsilon()
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

def read_for_training(p):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    return read_cropped_image(p, True)

def read_for_validation(p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return read_cropped_image(p, False )

if isfile('p2h.pickle'):
    with open('p2h.pickle', 'rb') as f:
        p2h = pickle.load(f)
else:
    # Compute phash for each image in the training and test set.
    p2h = {}
    for p in tqdm(join):
        img    = pil_image.open(expand_path(p))
        h      = phash(img)
        p2h[p] = h

    # Find all images associated with a given phash value.
    h2ps = {}
    for p,h in p2h.items():
        if h not in h2ps: h2ps[h] = []
        if p not in h2ps[h]: h2ps[h].append(p)

    # Find all distinct phash values
    hs = list(h2ps.keys())

    # If the images are close enough, associate the two phash values (this is the slow part: n^2 algorithm)
    h2h = {}
    for i,h1 in enumerate(tqdm(hs)):
        for h2 in hs[:i]:
            if h1-h2 <= 6 and match(h1, h2):
                s1 = str(h1)
                s2 = str(h2)
                if s1 < s2: s1,s2 = s2,s1
                h2h[s1] = s2

    # Group together images with equivalent phash, and replace by string format of phash (faster and more readable)
    for p,h in p2h.items():
        h = str(h)
        if h in h2h: h = h2h[h]
        p2h[p] = h
        
    pickle.dump(p2h, open('p2h.pickle', "wb")) 
    with open('p2h.pickle', 'wb') as handle:
        pickle.dump(p2h, handle)

#len(p2h), list(p2h.items())[:5]
        


h2ps = {}
for p,h in p2h.items():
    if h not in h2ps: h2ps[h] = []
    if p not in h2ps[h]: h2ps[h].append(p)
# Notice how 25460 images use only 20913 distinct image ids.
#len(h2ps),list(h2ps.items())[:5]
    


h2p = {}
for h,ps in h2ps.items(): h2p[h] = prefer(ps)
#len(h2p),list(h2p.items())[:5]

img_shape    = (384,384,1)
#img_shape    = (224,224, 3)

 # The margin added around the bounding box to compensate for bounding box inaccuracy

rotate = []




# Find all the whales associated with an image id. It can be ambiguous as duplicate images may have different whale ids.
h2ws = {}
new_whale = 'new_whale'
for p,w in tagged.items():
   # if w != new_whale: # Use only identified whales
        h = p2h[p]
        if h not in h2ws: h2ws[h] = []
        if w not in h2ws[h]: h2ws[h].append(w)
for h,ws in h2ws.items():
    if len(ws) > 1:
        h2ws[h] = sorted(ws)
#len(h2ws)
        
# For each whale, find the unambiguous images ids.
w2hs = {}
for h,ws in h2ws.items():
    if len(ws) == 1: # Use only unambiguous pictures
     #   if h2p[h] in exclude:
     #       print(h) # Skip excluded images
     #   else:
            w = ws[0]
            if w not in w2hs: w2hs[w] = []
            if h not in w2hs[w]: w2hs[w].append(h)
for w,hs in w2hs.items():
    if len(hs) > 1:
        w2hs[w] = sorted(hs)
#len(w2hs)


# Find the list of training images, keep only whales with at least two images.
train = [] # A list of training image ids
for hs in w2hs.values():
    if len(hs) > 1:
        train += hs
random.shuffle(train)
train_set = set(train)

w2ts = {} # Associate the image ids from train to each whale id.
#这里在match 列表中用到，去掉new_whale
for w,hs in w2hs.items():
    if w == new_whale:continue
    for h in hs:
        if h in train_set:
            if w not in w2ts: w2ts[w] = []
            if h not in w2ts[w]: w2ts[w].append(h)
for w,ts in w2ts.items(): w2ts[w] = np.array(ts)
    
t2i = {} # The position in train of each training image id
for i,t in enumerate(train): t2i[t] = i

#len(train),len(w2ts)

#from keras.utils import Sequence

# First try to use lapjv Linear Assignment Problem solver as it is much faster.
# At the time I am writing this, kaggle kernel with custom package fail to commit.
# scipy can be used as a fallback, but it is too slow to run this kernel under the time limit
# As a workaround, use scipy with data partitioning.
# Because algorithm is O(n^3), small partitions are much faster, but not what produced the submitted solution
try:
    from lap import lapjv
    segment = False
except ImportError:
    print('Module lap not found, emulating with much slower scipy.optimize.linear_sum_assignment')
    segment = True
    from scipy.optimize import linear_sum_assignment

class TrainingData(Sequence):
    def __init__(self, steps=1000, batch_size=32):
        """
        @param score the cost matrix for the picture matching
        @param steps the number of epoch we are planning with this score matrix
        """
        super(TrainingData, self).__init__()
      #  self.score      = -score # Maximizing the score is the same as minimuzing -score.
        self.steps      = steps
        self.batch_size = batch_size
        #同一类的样本，score初始化一个大值
     #   for ts in w2ts.values():
     #       idxs = [t2i[t] for t in ts]
     #       for i in idxs:
     ##              self.score[i,j] = 10000.0 # Set a large value for matching whales -- eliminates this potential pairing
        self.on_epoch_end()
    def __getitem__(self, index):
        start = self.batch_size*index
        end   = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size  = end - start
        assert size > 0
        a     = np.zeros((size,) + img_shape, dtype=K.floatx())
        b     = np.zeros((size,) + img_shape, dtype=K.floatx())
        c     = np.zeros((size,1), dtype=K.floatx())
        j     = start//2
        for i in range(0, size, 2):
            a[i,  :,:,:] = read_for_training(self.match[j][0])
            b[i,  :,:,:] = read_for_training(self.match[j][1])
            c[i,  0    ] = 1 # This is a match
            a[i+1,:,:,:] = read_for_training(self.unmatch[j][0])
            b[i+1,:,:,:] = read_for_training(self.unmatch[j][1])
            c[i+1,0    ] = 0 # Different whales
            j           += 1
        return [a,b],c
    def on_epoch_end(self):
        if self.steps <= 0: return # Skip this on the last epoch.
        self.steps     -= 1
        self.match      = []
        self.unmatch    = []
      #  if segment:
            # Using slow scipy. Make small batches.
            # Because algorithm is O(n^3), small batches are much faster.
            # However, this does not find the real optimum, just an approximation.
       ##     batch = 512
       #     for start in range(0, score.shape[0], batch):
       #         end = min(score.shape[0], start + batch)
       #         _, x = linear_sum_assignment(self.score[start:end, start:end])
       #         tmp.append(x + start)
       #     x = np.concatenate(tmp)
       # else:
       #     _,_,x = lapjv(self.score) # Solve the linear assignment problem
       # y = np.arange(len(x),dtype=np.int32)

        # Compute a derangement for matching whales
        # self.match :所有同类的两两组合[(ts1, ts2), (ts3, ts4)]
        for ts in w2ts.values():
            d = ts.copy()
            while True:
                random.shuffle(d)
                if not np.any(ts == d): break
            for ab in zip(ts,d): self.match.append(ab)
        
        match_len = len(self.match)
    #    print('len ', len(train))
        
       # print(i_list)
        selected_list = []
        while len(self.match) > len(self.unmatch):
          i_list = list(range(len(train)))
          random.shuffle(i_list)
          for i in tqdm(range(match_len)):
            m1 = 0
            m2 = 0
            
            for train_i in i_list: 
             #   if train_i not in selected_list:
                    m1 = train[train_i]
                 #   print('t_i ', train_i, ' m ', m1)
                   # selected_list.append(train_i)
                   # t_i = train_i
                    i_list.remove(train_i)
                    break
            
         #   print('m1 ', m1)
            if m1 == 0:continue
            m1ws = h2ws[m1]
         #   print('ws ', m1ws)
            for train_i in i_list:
              #  if train_i not in selected_list:
                    h = train[train_i]
                    ws = h2ws[h]
                    if_same_w = False
                    for w in ws:
                        if w in m1ws:
                            if_same_w = True
                            break
                    if if_same_w == False:
                        m2 = h
                       # selected_list.append(train_i)
                        i_list.remove(train_i)
                        break
                    
            if m1 != 0 and m2 != 0:
                self.unmatch.append((m1, m2))
            if len(self.match) == len(self.unmatch):
                break
            if len(i_list) == 0:
                i_list = list(range(len(train)))
                random.shuffle(i_list)
                        
                
        # Construct unmatched whale pairs from the LAP solution.
        # self.unmatch :不同类样本的ts两两组合列表[(tsu1, tsu2), (tsu3, tsu4)],x是从匈牙利算法得出的基于score矩阵的结果序列
      #      if i == j:
      #          print(self.score)
      #          print(x)
      #          print(y)
      #          print(i,j)
     #       assert i != j
      #      self.unmatch.append((train[i],train[j]))

        # Force a different choice for an eventual next epoch.
     #   self.score[x,y] = 10000.0
     #   self.score[y,x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        
        print(len(self.match), len(train), len(self.unmatch))
     #   assert len(self.match) == len(train) and len(self.unmatch) == len(train)
    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1)//self.batch_size

# Test on a batch of 32 with random costs.
#score = np.random.random_sample(size=(len(train),len(train)))
#data = TrainingData(score)
#(a, b), c = data[0]
#a.shape, b.shape, c.shape
# A Keras generator to evaluate only the BRANCH MODEL
class FeatureGen(Sequence):
    def __init__(self, data,img_shape, batch_size=64, verbose=1):
        super(FeatureGen, self).__init__()
        self.data       = data
        self.batch_size = batch_size
        self.verbose    = verbose
        self.img_shape = img_shape
    
        if self.verbose > 0: self.progress = tqdm(total=len(self), desc='Features')
    def __getitem__(self, index):
        start = self.batch_size*index
        size  = min(len(self.data) - start, self.batch_size)
        a     = np.zeros((size,) + self.img_shape, dtype=K.floatx())
        for i in range(size): a[i,:,:,:] = read_for_validation(self.data[start + i])
        if self.verbose > 0: 
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return a
    def __len__(self):
        return (len(self.data) + self.batch_size - 1)//self.batch_size
from keras_tqdm import TQDMCallback

from keras_model import build_model


model, branch_model, head_model = build_model(64e-5,0, img_shape)
head_model.summary()
branch_model.summary()
model.summary()

plot_model(head_model, to_file='head-model.png')
pil_image.open('head-model.png')

plot_model(branch_model, to_file='branch_model.png')
pil_image.open('branch_model.png')

plot_model(model, to_file='model.png')
pil_image.open('model.png')

# A Keras generator to evaluate on the HEAD MODEL on features already pre-computed.
# It computes only the upper triangular matrix of the cost matrix if y is None.
class ScoreGen(Sequence):
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
def set_lr(model, lr):
    K.set_value(model.optimizer.lr, float(lr))

def get_lr(model):
    return K.get_value(model.optimizer.lr)
def save_model(steps):
    branch_model.save(str(steps) + 'mpiotte-standard_branch.model')
    head_model.save(str(steps) + 'mpiotte-standard_header.model')  
    model.save(str(steps) + 'mpiotte-standard_model.model')
   
    branch_model_json = branch_model.to_json()
    with open(str(steps) + "branch_model.json", "w") as json_file:
        json_file.write(branch_model_json)
    branch_model.save_weights(str(steps) + "branch_model.h5") 

    head_model_json = head_model.to_json()
    with open(str(steps) + "head_model.json", "w") as json_file:
        json_file.write(head_model_json)
    head_model.save_weights(str(steps) + "head_model.h5") 

    model_json = model.to_json()
    with open(str(steps) + "main_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(str(steps) + "main_model.h5") 


#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
#loaded_model.load_weights("model.h5")
# Not computing the submission in this notebook because it is a little slow. It takes about 15 minutes on setup with a GTX 1080.
import gzip

def prepare_submission(known,score, threshold, filename):
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
        for i,p in enumerate(tqdm(submit)):
            t = []
            s = set()
            a = score[i,:]
            for j in list(reversed(np.argsort(a))):
                h = known[j]
                if a[j] < threshold and new_whale not in s:
                    pos[len(t)] += 1
                    s.add(new_whale)
                    t.append(new_whale)
                    if len(t) == 5: break;
                for w in h2ws[h]:
                    assert w != new_whale
                    if w not in s:
                        if a[j] > 1.0:
                            vtop += 1
                        elif a[j] >= threshold:
                            vhigh += 1
                        s.add(w)
                        t.append(w)
                        if len(t) == 5: break;
                if len(t) == 5: break;
            if new_whale not in s: pos[5] += 1
            assert len(t) == 5 and len(s) == 5
            f.write(p + ',' + ' '.join(t[:5]) + '\n')
    return vtop,vhigh,pos

def genSubFile(steps):
    # Find elements from training sets not 'new_whale'
    h2ws = {}
    for p,w in tagged.items():
        if w != new_whale: # Use only identified whales
            h = p2h[p]
            if h not in h2ws: h2ws[h] = []
            if w not in h2ws[h]: h2ws[h].append(w)
    known = sorted(list(h2ws.keys()))

    # Dictionary of picture indices
    h2i   = {}
    for i,h in enumerate(known): h2i[h] = i

    # Evaluate the model.
    fknown  = branch_model.predict_generator(FeatureGen(known, img_shape), max_queue_size=20, workers=10, verbose=0)
    fsubmit = branch_model.predict_generator(FeatureGen(submit, img_shape), max_queue_size=20, workers=10, verbose=0)
    score   = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
    score   = score_reshape(score, fknown, fsubmit)

    # Generate the subsmission file.
    prepare_submission(known,score, 0.99, str(steps) + '_zl_mpiotte-standard_mine.csv.gz')
    

steps = 0
sub_index_list = [0, 140, 160, 220, 320, 400]

class LossHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        writer.add_scalar('Train/loc_loss', logs.get('loss'), epoch)
        writer.add_scalar('Train/acc', logs.get('acc'), epoch)
       # steps += 1
        if epoch in sub_index_list:
            save_model(epoch)
            genSubFile(epoch)

        
def score_reshape(score, x, y=None):
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
        m = np.zeros((x.shape[0],x.shape[0]), dtype=K.floatx())
        m[np.triu_indices(x.shape[0],1)] = score.squeeze()
        m += m.transpose()
    else:
        m        = np.zeros((y.shape[0],x.shape[0]), dtype=K.floatx())
        iy,ix    = np.indices((y.shape[0],x.shape[0]))
        ix       = ix.reshape((ix.size,))
        iy       = iy.reshape((iy.size,))
        m[iy,ix] = score.squeeze()
    return m

def compute_score(verbose=1):
    """
    Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
    """
    features = branch_model.predict_generator(FeatureGen(train, img_shape,verbose=verbose), max_queue_size=12, workers=6, verbose=0)
    score    = head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=6, verbose=0)
    score    = score_reshape(score, features)
    return features, score
import cv2
def to_grayscale(img):
    """
    input is (d,w,h)
    converts 3D image tensor to grayscale images corresponding to each channel
    """
    # print(image.shape)
    channel = img.shape[0]
    img = torch.sum(img, dim=0)
    # print(image.shape)
    img = torch.div(img, channel)
    # print(image.shape)
    # assert False
    return img
def make_steps(step, ampl):
    """
    Perform training epochs
    @param step Number of epochs to perform
    @param ampl the K, the randomized component of the score matrix.
    """
    global w2ts, t2i, features,steps, score, histories
    
    
    # shuffle the training pictures
    #train 是w--hs list,过滤了只有一个样本的w
    random.shuffle(train)
    
    # Map whale id to the list of associated training picture hash value
    w2ts = {}
    for w,hs in w2hs.items():
        if w == new_whale:continue
        for h in hs:
            if h in train_set:
                if w not in w2ts: w2ts[w] = []
                if h not in w2ts[w]: w2ts[w].append(h)
    for w,ts in w2ts.items(): w2ts[w] = np.array(ts)

    # Map training picture hash value to index in 'train' array    
    t2i  = {}
    for i,t in enumerate(train): t2i[t] = i    

    # Compute the match score for each picture pair
  #  features, score = compute_score()
    
 #   for i in [0]:
 #       feature_map = features[i]
      #  print('feature_map ', feature_map.shape)
       # feature_map = feature_map.squeeze(0)
       # temp = to_grayscale(feature_map).data.cpu().numpy()
 #       feature_map = feature_map * 255
  #      feature_heatmap = cv2.applyColorMap(feature_map.astype(np.uint8), cv2.COLORMAP_JET)
 #       writer.add_image('{}test/{}'.format(i, steps), feature_heatmap, steps)
    
    # Train the model for 'step' epochs
    history = model.fit_generator(
        TrainingData(steps=step, batch_size=32),
        initial_epoch=steps, epochs=steps + step, max_queue_size=12, workers=6, verbose=0,
        callbacks=[
            TQDMCallback(leave_inner=True, metric_format='{value:0.3f}'),
            LossHistory()
        ]).history
    
  #  for i in [0, 100, 1000, 130]:
  #      feature_map = features[i]
      #  print('feature_map ', feature_map.shape)
     #   feature_map = feature_map.squeeze(0)
     #   temp = to_grayscale(feature_map).data.cpu().numpy()
      #  temp = to_grayscale(feature_map).data.cpu().numpy()

  #      feature_map = feature_map * 255
  #      feature_heatmap = cv2.applyColorMap(feature_map.astype(np.uint8), cv2.COLORMAP_JET)
  #      writer.add_image('{}/{}'.format(i, steps), feature_heatmap, steps)
    
    steps += step
    
    # Collect history data
    history['epochs'] = steps
 #s   history['ms'    ] = np.mean(score)
    history['lr'    ] = get_lr(model)
    print(history['epochs'],history['lr'])
    histories.append(history)
    
    
        
    

model_name = 'mpiotte-standard-'
histories  = []
#steps      = 0

if False:
    tmp = keras.models.load_model('mpiotte-standard.model')
    model.set_weights(tmp.get_weights())
else:
    # epoch -> 10
   # make_steps(1, 1000)
  #  model.save(model_name)#test
    

    make_steps(10, 1000)
    ampl = 100.0
    for _ in range(10):
        print('noise ampl.  = ', ampl)
        make_steps(5, ampl)
        ampl = max(1.0, 100**-0.1*ampl)
    # epoch -> 150
    for _ in range(18): make_steps(5, 1.0)
    # epoch -> 200
    set_lr(model, 16e-5)
    for _ in range(10): make_steps(5, 0.5)
    # epoch -> 240
    set_lr(model, 4e-5)
    for _ in range(8): make_steps(5, 0.25)
    # epoch -> 250
    set_lr(model, 1e-5)
    for _ in range(2): make_steps(5, 0.25)
    # epoch -> 300
   # weights = model.get_weights()
   # model, branch_model, head_model = build_model(64e-5,0.0002)
   # model.set_weights(weights)
    for _ in range(10): make_steps(5, 1.0)
    # epoch -> 350
    set_lr(model, 16e-5)
    for _ in range(10): make_steps(5, 0.5)    
    # epoch -> 390
    set_lr(model, 4e-5)
    for _ in range(8): make_steps(5, 0.25)
    # epoch -> 400
    set_lr(model, 1e-5)
    for _ in range(2): make_steps(5, 0.25)
    
    

    
    
    
    
    