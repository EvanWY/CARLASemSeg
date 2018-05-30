from __future__ import print_function
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
import argparse
import logging
import random
import time

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

client = None

def init_sim_connect():
    global client
    print ('initializing CARLA client connection')
    with make_carla_client('localhost', 2000) as carla_client:
        print('CarlaClient connected !')
        client = carla_client
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=20,
            NumberOfPedestrians=40,
            WeatherId=random.choice([1, 3, 7, 8, 14]),
            QualityLevel='Epic')
        settings.randomize_seeds()
        settings.randomize_weather()

        camera0 = Camera('CameraRGB')
        camera0.set_image_size(800, 600)
        camera0.set_position(0.30, 0, 1.30)
        settings.add_sensor(camera0)

        camera1 = Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
        camera1.set_image_size(800, 600)
        camera1.set_position(0.30, 0, 1.30)
        settings.add_sensor(camera1)
        print ('before load setting')
        scene = client.load_settings(settings)
        print ('after load setting')

frame_count = 300
def get_next_frame():
    global frame_count
    global client
    if client is None:
        init_sim_connect()
    frame_count += 1
    if frame_count >= 300:
        frame_count = 0
        #init
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=20,
            NumberOfPedestrians=40,
            WeatherId=random.choice([1, 3, 7, 8, 14]),
            QualityLevel='Epic')
        settings.randomize_seeds()
        settings.randomize_weather()

        camera0 = Camera('CameraRGB')
        camera0.set_image_size(800, 600)
        camera0.set_position(0.30, 0, 1.30)
        settings.add_sensor(camera0)

        camera1 = Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
        camera1.set_image_size(800, 600)
        camera1.set_position(0.30, 0, 1.30)
        settings.add_sensor(camera1)
        print ('before load setting')
        scene = client.load_settings(settings)
        print ('after load setting')

        number_of_player_starts = len(scene.player_start_spots)
        player_start = random.randint(0, max(0, number_of_player_starts - 1))

        print('Starting new episode at %r...' % scene.map_name)
        client.start_episode(player_start)

    measurements, sensor_data = client.read_data()
    for name, measurement in sensor_data.items():
        if name == 'CameraRGB':
            img = measurement.raw_data
        elif name == 'CameraSemSeg':
            seg = measurement.raw_data
        else:
            print ('sensor name incorrect: %s'%name)
            exit()

    control = measurements.player_measurements.autopilot_control
    control.steer += random.uniform(-0.1, 0.1)
    client.send_control(control)

    return img, seg

    

def zerg_generator(samples, batch_size=20):
    while 1:
        img_list = []
        seg_list = []
        for batch_id in range(batch_size):

            #img = cv2.imread(batch_sample[0])
            #seg = cv2.imread(batch_sample[1])

            print ('before hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
            img, seg = get_next_frame()
            print ('after hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
            print (img.shape)
            print (seg.shape)
            cv2.imwrite('tttttttttest.png', img)
            exit()

            temp_ = seg[496:600,:,:]
            temp_ = (temp_ != 10) * temp_
            seg[496:600,:,:] = temp_

            t = 600 - random.randint(0,12)
            b = 0 + random.randint(0,12)
            r = 800 - random.randint(0,16)
            l = 0 + random.randint(0,16)
            
            img = img[b:t, l:r]
            seg = seg[b:t, l:r]

            img = cv2.resize(img, (320, 320), interpolation = cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            seg = cv2.resize(seg, (320, 320), interpolation = cv2.INTER_NEAREST)[:,:,2]
            seg_road = np.logical_or(seg == 7 ,seg == 6).astype(np.uint8)
            seg_vehicle = (seg == 10).astype(np.uint8)
            seg = np.zeros((320, 320, 2)).astype(np.uint8)
            seg [:,:,0] = seg_road
            seg [:,:,1] = seg_vehicle

            img_list.append(img)
            seg_list.append(seg)

        # trim image to only see section with road
        X_train = np.array(img_list).reshape(batch_size, 320, 320, 3)
        y_train = np.array(seg_list).reshape(batch_size, 320, 320, 2)
        yield sklearn.utils.shuffle(X_train, y_train)

class FitGenCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        img = cv2.resize(cv2.imread('visualize_imgs/rgb.png'), (320, 320), interpolation = cv2.INTER_CUBIC)
        visualization_img = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        z = np.zeros([20,320,320,3])
        z[0,:,:,:] = img
        seg = self.model.predict(z)[0,:,:,:]
        seg = seg.reshape(320,320,2)
        seg_road = (seg[:,:,0] > 0.5).astype(np.uint8) * 127
        seg_vehicle = (seg[:,:,1] > 0.5).astype(np.uint8) * 127
        
        visualization_img = visualization_img // 2
        visualization_img[:,:,0] += seg_road
        visualization_img[:,:,1] += seg_vehicle

        cv2.imwrite('visualize_imgs/seg-epoch_%03d.png' % epoch, visualization_img)

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
    #model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    model.compile(loss = 'mse', optimizer = 'adam')

    model.fit_generator(
        train_generator,
        steps_per_epoch = 45, 
        epochs = 1720,
        validation_data = validation_generator, 
        validation_steps = 5,
        callbacks = [FitGenCallback()]
    )

    model.save('zerg_model_%s.h5'%datetime.datetime.now().strftime("%Y%m%d+%H%M%S"))
    exit()
