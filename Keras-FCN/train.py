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
from subprocess import call

def zerg_generator(samples, batch_size=20):
    print ('zerg_generator is called')
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples+1-batch_size, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            img_list = []
            seg_list = []
            for batch_sample in batch_samples:
                img = cv2.imread(batch_sample[0])

                seg = cv2.imread(batch_sample[1])
                temp_ = seg[496:600,:,:]
                temp_ = (temp_ != 10) * temp_
                seg[496:600,:,:] = temp_

                # t = 600 - random.randint(0,12)
                # b = 0 + random.randint(0,12)
                # r = 800 - random.randint(0,16)
                # l = 0 + random.randint(0,16)
                
                # img = img[b:t, l:r]
                # seg = seg[b:t, l:r]
                
                img = cv2.resize(img, (320, 320), interpolation = cv2.INTER_CUBIC)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                seg = cv2.resize(seg, (320, 320), interpolation = cv2.INTER_NEAREST)[:,:,2]
                seg_road = np.logical_or(seg == 7 ,seg == 6).astype(np.uint8)
                seg_vehicle = (seg == 10).astype(np.uint8)
                seg = np.zeros((320, 320, 2)).astype(np.uint8)
                seg [:,:,0] = seg_road
                seg [:,:,1] = seg_vehicle

                # img_flip = cv2.flip(img, 1)
                # seg_flip = cv2.flip(seg, 1)

                img_list.append(img)
                seg_list.append(seg)
                # img_list.append(img_flip)
                # seg_list.append(seg_flip)

            # trim image to only see section with road
            X_train = np.array(img_list).reshape(-1,320, 320, 3)
            y_train = np.array(seg_list).reshape(-1,320, 320, 2)
            yield sklearn.utils.shuffle(X_train, y_train)

class FitGenCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        for idx in range(10):
            try:
                img = cv2.resize(cv2.imread('visualize_imgs/rgb{0:03d}.png'.format(idx)), (320, 320), interpolation = cv2.INTER_CUBIC)
                visualization_img = img
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                z = np.zeros([20,320,320,3])
                z[0,:,:,:] = img
                
                seg = self.model.predict(z)[0,:,:,:]
                seg = seg.reshape(320,320,2)
                seg_road = (seg[:,:,0] > 0.5).astype(np.uint8) * 127
                seg_vehicle = (seg[:,:,1] > 0.5).astype(np.uint8) * 127
                
                visualization_img = visualization_img // 2
                visualization_img[:,:,0] += seg_road
                visualization_img[:,:,1] += seg_vehicle

                cv2.imwrite('visualize_imgs/seg-epoch{0:03d}-{1:03d}.png'.format(epoch, idx), visualization_img)
            except:
                print ("Unexpected error:" + sys.exc_info()[0])

        model_name = 'terran_model_{0}_epoch{1:03d}.h5'.format(datetime.datetime.now().strftime("%Y%m%d+%H%M%S"), epoch)
        self.model.save(model_name)
        call(['aws', 's3', 'cp', model_name, 's3://yang-carla-train'])

        return

if __name__ == '__main__':
    samples = []
    for line in range(120000):
        samples.append(['../Train/CameraRGB/%07d.png' % line, '../Train/CameraSeg/%07d.png' % line])

    train_samples, validation_samples = train_test_split(samples, test_size=0.10)
    # compile and train the model using the generator function
    train_generator = zerg_generator(train_samples, batch_size=20)
    validation_generator = zerg_generator(validation_samples, batch_size=20)

    model = zerg_model(batch_shape=[20, 320, 320, 3])

    train_mode = sys.argv[-1]
    if train_mode == 'resume':
        model.load_weights('terran_model.h5')
    elif train_mode == 'new':
        pass
    else:
        print ('specify training mode, `python train.py resume` or `python train.py new`')
        exit()

    model.summary()
    print('### train sample size == {}, validation sample size == {}'.format(len(train_samples), len(validation_samples)))
    #model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    model.compile(loss = 'mse', optimizer = 'adam')

    model.fit_generator(
        train_generator,
        steps_per_epoch = 500, 
        epochs = 300,
        validation_data = validation_generator, 
        validation_steps = 50,
        callbacks = [FitGenCallback()]
    )

    model_name = 'terran_model_%s.h5'%datetime.datetime.now().strftime("%Y%m%d+%H%M%S")
    model.save(model_name)
    call(['aws', 's3', 'cp', model_name, 's3://yang-carla-train'])
    exit()