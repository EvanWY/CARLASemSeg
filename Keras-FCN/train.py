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
from sklearn.model_selection import train_test_split

def zerg_generator(samples, batch_size=36):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size // 2):
            batch_samples = samples[offset:offset + batch_size // 2]

            rgb_imgs = []
            seg_imgs = []
            for batch_sample in batch_samples:
                rgb = cv2.resize(cv2.imread(batch_sample[0]), (320, 320), interpolation = cv2.INTER_CUBIC)
                
                raw_seg = cv2.resize(cv2.imread(batch_sample[1]), (320, 320), interpolation = cv2.INTER_NEAREST)[:,:,2]
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

def zerg_model(weight_decay=0., batch_momentum=0.9, batch_shape=[36, 320, 320, 3], classes=2):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        print ('zerg_model should have batch_shape')
        exit()

    bn_axis = 3

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', weight_decay=weight_decay, strides=(1, 1), batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f', weight_decay=weight_decay, batch_momentum=batch_momentum)(x)

    x = atrous_conv_block(3, [512, 512, 2048], stage=5, block='a', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    x = atrous_identity_block(3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay, atrous_rate=(2, 2), batch_momentum=batch_momentum)(x)
    #classifying layer
    #x = Conv2D(classes, (3, 3), dilation_rate=(2, 2), kernel_initializer='normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='same', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)
    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)
    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
    model.load_weights(weights_path, by_name=True)
    return model

if __name__ == '__main__':
    samples = []
    for line in range(1000):
        samples.append(['../Train/CameraRGB/%d.png' % line, '../Train/CameraSeg/%d.png' % line])

    train_samples, validation_samples = train_test_split(samples, test_size=0.15)
    # compile and train the model using the generator function
    train_generator = zerg_generator(train_samples, batch_size=36)
    validation_generator = zerg_generator(validation_samples, batch_size=36)

    model = zerg_model()

    model.summary()
    print('### train sample size == {}, validation sample size == {}'.format(len(train_samples), len(validation_samples)))
    model.compile(loss = 'mse', optimizer = 'adam')
    model.fit_generator(
        train_generator,
        samples_per_epoch = len(train_samples) * 2, 
        validation_data = validation_generator, 
        nb_val_samples = len(validation_samples) * 2, 
        nb_epoch = 12)

    model.save('zerg_model.h5')
    exit()