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
    if (id % 10 == 0):
        print ('processing .. {0}/{1}'.format(id, frames))
    img = cv2.imread('../Train/CameraRGB/%d.png' % id)
    img = cv2.resize(img, (320, 320), interpolation = cv2.INTER_CUBIC)
    visualization_img = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    seg = model.predict(img.reshape(1,320,320,3))
    seg = seg.reshape(320,320,2)

    if 'road_heat' == sys.argv[-1]:
        pass
    elif 'vehicle_heat' == sys.argv[-1]:
        heat = seg[:,:,1].clip(0,1)
        R = heat.clip(0,0.3333) * 3
        G = (heat.clip(0.3334, 0.6666) - 0.3333) * 3
        B = (heat.clip(0.6667, 1) - 0.6666) * 3
        visualization_img = visualization_img * 0
        visualization_img[:,:,0] = (B * 255).astype(np.uint8)
        visualization_img[:,:,1] = (G * 255).astype(np.uint8)
        visualization_img[:,:,2] = (R * 255).astype(np.uint8)
    else
        seg_road = (seg[:,:,0] > 0.5).astype(np.uint8) * 127
        seg_vehicle = (seg[:,:,1] > 0.5).astype(np.uint8) * 127
        
        visualization_img = visualization_img // 2
        visualization_img[:,:,0] += seg_road
        visualization_img[:,:,1] += seg_vehicle
        
    visualization_img = cv2.resize(visualization_img, (800, 600), interpolation = cv2.INTER_CUBIC)
    # cv2.imwrite('visualize_imgs/seg.png', rgb_fullsize, interpolation = cv2.INTER_NEAREST))
    out_video[id,:,:,:] = visualization_img

skvideo.io.vwrite('visualize_imgs/seg.mp4', out_video)