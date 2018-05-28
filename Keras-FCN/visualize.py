import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import pickle
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import *
from keras.objectives import *
from keras.metrics import binary_accuracy
from keras.models import load_model
import keras.backend as K
#import keras.utils.visualize_util as vis_util

from models import *
from utils.loss_function import *
from utils.metrics import *
from utils.SegDataGenerator import *
import time
import sklearn
from sklearn.model_selection import train_test_split
import sys, skvideo.io

# file = sys.argv[-1]
# if file == 'visualize.py':
#     print ("Error loading video")
#     quit

# video = skvideo.io.vread(file)
# out_video = np.zeros_like(video)

frames = 1000

model = zerg_model(batch_shape=[1, 320, 320, 3])
model.load_weights('zerg_model.h5')

out_video = np.zeros([frames, 600, 800, 3])

for id in range(frames):
    rgb_frame = cv2.imread('../Train/CameraRGB/%d.png' % id)
    rgb = cv2.resize(rgb_frame, (320, 320), interpolation = cv2.INTER_CUBIC)

    seg = model.predict(rgb.reshape(1,320,320,3))
    seg = seg.reshape(320,320,2)
    seg_road = (seg[:,:,0] > 0.5).astype(np.uint8) * 127
    seg_vehicle = (seg[:,:,1] > 0.5).astype(np.uint8) * 127
    
    rgb = rgb // 2
    rgb[:,:,0] += seg_road
    rgb[:,:,1] += seg_vehicle
    
    rgb_fullsize = cv2.resize(rgb, (800, 600), interpolation = cv2.INTER_CUBIC)
    # cv2.imwrite('visualize_imgs/seg.png', rgb_fullsize, interpolation = cv2.INTER_NEAREST))
    out_video[id,:,:,:] = rgb_fullsize

skvideo.io.vwrite('visualize_imgs/seg.mp4', out_video)