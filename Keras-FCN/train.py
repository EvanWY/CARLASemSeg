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
import keras
import datetime
import random

def zerg_generator(samples, batch_size=20):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size // 2):
            batch_samples = samples[offset:offset + batch_size // 2]

            rgb_imgs = []
            seg_imgs = []
            for batch_sample in batch_samples:
                rgb_fullsize = cv2.imread(batch_sample[0])
                raw_seg_fullsize = cv2.imread(batch_sample[1])
                raw_seg_fullsize[496:600,:,:] = 0

                t = 600 - random.randint(0,12)
                b = 0 + random.randint(0,12)
                r = 800 - random.randint(0,16)
                l = 0 + random.randint(0,16)
                rgb_crop = rgb_fullsize[b:t, l:r]
                raw_seg_crop = raw_seg_fullsize[b:t, l:r]
                

                rgb = cv2.resize(rgb_crop, (320, 320), interpolation = cv2.INTER_CUBIC)
                raw_seg = cv2.resize(raw_seg_crop, (320, 320), interpolation = cv2.INTER_NEAREST)[:,:,2]
                seg_road = np.logical_or(raw_seg == 7 ,raw_seg == 6).astype(np.uint8)
                seg_vehicle = (raw_seg == 10).astype(np.uint8)
                seg = np.zeros((320, 320, 2)).astype(np.uint8)
                seg [:,:,0] = seg_road
                seg [:,:,1] = seg_vehicle

                #rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                rgb_flip = cv2.flip(rgb, 1)
                seg_flip = cv2.flip(seg, 1)

                rgb_imgs.append(rgb)
                seg_imgs.append(seg)
                rgb_imgs.append(rgb_flip)
                seg_imgs.append(seg_flip)

            # trim image to only see section with road
            X_train = np.array(rgb_imgs).reshape(-1,320, 320, 3)
            y_train = np.array(seg_imgs).reshape(-1,320, 320, 2)
            yield sklearn.utils.shuffle(X_train, y_train)

class FitGenCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        rgb = cv2.resize(cv2.imread('visualize_imgs/rgb.png'), (320, 320), interpolation = cv2.INTER_CUBIC)
        
        z = np.zeros([20,320,320,3])
        z[0,:,:,:] = rgb
        seg = self.model.predict(z)[0,:,:,:]
        seg = seg.reshape(320,320,2)
        seg_road = (seg[:,:,0] > 0.5).astype(np.uint8) * 127
        seg_vehicle = (seg[:,:,1] > 0.5).astype(np.uint8) * 127
        
        rgb = rgb // 2
        rgb[:,:,0] += seg_road
        rgb[:,:,1] += seg_vehicle

        cv2.imwrite('visualize_imgs/seg-epoch_%03d.png' % epoch, rgb)

        if (epoch % 20 == 1):
            self.model.save('zerg_model_{0}_epoch{1:03d}.h5'.format(datetime.datetime.now().strftime("%Y%m%d+%H%M%S"), epoch))

        return

if __name__ == '__main__':
    samples = []
    for line in range(1000):
        samples.append(['../Train/CameraRGB/%d.png' % line, '../Train/CameraSeg/%d.png' % line])

    train_samples, validation_samples = train_test_split(samples, test_size=0.10)
    # compile and train the model using the generator function
    train_generator = zerg_generator(train_samples, batch_size=20)
    validation_generator = zerg_generator(validation_samples, batch_size=20)

    model = zerg_model(batch_shape=[20, 320, 320, 3])

    train_mode = sys.argv[-1]
    if train_mode == 'resume':
        model.load_weights('zerg_model.h5')
    elif train_mode == 'new':
        pass
    else:
        print ('specify training mode, `python train.py resume` or `python train.py new`')
        exit()

    model.summary()
    print('### train sample size == {}, validation sample size == {}'.format(len(train_samples), len(validation_samples)))
    model.compile(loss = 'mse', optimizer = 'adam')

    model.fit_generator(
        train_generator,
        steps_per_epoch = 45, 
        epochs = 172,
        validation_data = validation_generator, 
        validation_steps = 5,
        callbacks = [FitGenCallback()]
    )

    model.save('zerg_model_%s.h5'%datetime.datetime.now().strftime("%Y%m%d+%H%M%S"))
    exit()