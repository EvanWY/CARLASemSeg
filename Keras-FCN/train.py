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
'''
def train(batch_size, epochs, lr_base, lr_power, weight_decay, classes,
          model_name, train_file_path, val_file_path,
          data_dir, label_dir, target_size=None, batchnorm_momentum=0.9,
          resume_training=False, class_weight=None, dataset='VOC2012',
          loss_fn=softmax_sparse_crossentropy_ignoring_last_label,
          metrics=[sparse_accuracy_ignoring_last_label],
          loss_shape=None,
          label_suffix='.png',
          data_suffix='.jpg',
          ignore_label=255,
          label_cval=255):
    if target_size:
        input_shape = target_size + (3,)
    else:
        input_shape = (None, None, 3)
    batch_shape = (batch_size,) + input_shape

    ###########################################################
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, 'Models/' + model_name)
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)

    # ###############learning rate scheduler####################
    def lr_scheduler(epoch, mode='power_decay'):
        '''if epoch in lr_dict:
            lr = lr_dict[epoch]
            print 'lr: %f' % lr'''

        if mode is 'power_decay':
            # original lr scheduler
            lr = lr_base * ((1 - float(epoch)/epochs) ** lr_power)
        if mode is 'exp_decay':
            # exponential decay
            lr = (float(lr_base) ** float(lr_power)) ** float(epoch+1)
        # adam default lr
        if mode is 'adam':
            lr = 0.001

        if mode is 'progressive_drops':
            # drops as progression proceeds, good for sgd
            if epoch > 0.9 * epochs:
                lr = 0.0001
            elif epoch > 0.75 * epochs:
                lr = 0.001
            elif epoch > 0.5 * epochs:
                lr = 0.01
            else:
                lr = 0.1

        print('lr: %f' % lr)
        return lr
    scheduler = LearningRateScheduler(lr_scheduler)

    # ###################### make model ########################
    checkpoint_path = os.path.join(save_path, 'checkpoint_weights.hdf5')

    model = globals()[model_name](weight_decay=weight_decay,
                                  input_shape=input_shape,
                                  batch_momentum=batchnorm_momentum,
                                  classes=classes)

    # ###################### optimizer ########################
    optimizer = SGD(lr=lr_base, momentum=0.9)
    # optimizer = Nadam(lr=lr_base, beta_1 = 0.825, beta_2 = 0.99685)

    print ('######################  model.compile  ######################')
    model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=metrics)
    print ('######################  if resume_training:  ######################')
    if resume_training:
        model.load_weights(checkpoint_path, by_name=True)
    model_path = os.path.join(save_path, "model.json")
    # save model structure
    f = open(model_path, 'w')
    model_json = model.to_json()
    f.write(model_json)
    f.close
    img_path = os.path.join(save_path, "model.png")
    # #vis_util.plot(model, to_file=img_path, show_shapes=True)
    model.summary()

    # lr_reducer      = ReduceLROnPlateau(monitor=softmax_sparse_crossentropy_ignoring_last_label, factor=np.sqrt(0.1),
    #                                     cooldown=0, patience=15, min_lr=0.5e-6)
    # early_stopper   = EarlyStopping(monitor=sparse_accuracy_ignoring_last_label, min_delta=0.0001, patience=70)
    # callbacks = [early_stopper, lr_reducer]
    callbacks = [scheduler]

    # ####################### tfboard ###########################
    if K.backend() == 'tensorflow':
        tensorboard = TensorBoard(log_dir=os.path.join(save_path, 'logs'), histogram_freq=10, write_graph=True)
        callbacks.append(tensorboard)
    # ################### checkpoint saver#######################
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'checkpoint_weights.hdf5'), save_weights_only=True)#.{epoch:d}
    callbacks.append(checkpoint)
    # set data generator and train
    print ('######################  SegDataGenerator  ######################')
    train_datagen = SegDataGenerator(zoom_range=[0.5, 2.0],
                                     zoom_maintain_shape=True,
                                     crop_mode='random',
                                     crop_size=target_size,
                                     # pad_size=(505, 505),
                                     rotation_range=0.,
                                     shear_range=0,
                                     horizontal_flip=True,
                                     channel_shift_range=20.,
                                     fill_mode='constant',
                                     label_cval=label_cval)
    val_datagen = SegDataGenerator()

    print ('######################  def get_file_len  ######################')
    def get_file_len(file_path):
        fp = open(file_path)
        lines = fp.readlines()
        fp.close()
        return len(lines)

    # from Keras documentation: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished
    # and starting the next epoch. It should typically be equal to the number of unique samples of your dataset divided by the batch size.
    
    print ('######################  steps_per_epoch  ######################')
    steps_per_epoch = int(np.ceil(get_file_len(train_file_path) / float(batch_size)))


    print ('######################  generator  ######################')
    print ('train_file_path = %s' % train_file_path)
    print ('data_dir = %s' % data_dir)
    print ('data_suffix = %s' % data_suffix)
    print ('label_dir = %s' % label_dir)
    print ('label_suffix = %s' % label_suffix)
    generator=train_datagen.flow_from_directory(
        file_path=train_file_path,
        data_dir=data_dir, data_suffix=data_suffix,
        label_dir=label_dir, label_suffix=label_suffix,
        classes=classes,
        target_size=target_size, color_mode='rgb',
        batch_size=batch_size, shuffle=True,
        loss_shape=loss_shape,
        ignore_label=ignore_label,
        # save_to_dir='Images/'
    )

    print ('######################  history v2 ######################')
    print (epochs)
    print (steps_per_epoch)
    history = model.fit_generator(
        generator = generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        workers=4,
        # validation_data=val_datagen.flow_from_directory(
        #     file_path=val_file_path, data_dir=data_dir, data_suffix='.jpg',
        #     label_dir=label_dir, label_suffix='.png',classes=classes,
        #     target_size=target_size, color_mode='rgb',
        #     batch_size=batch_size, shuffle=False
        # ),
        # nb_val_samples = 64
        #class_weight=class_weight
    )

    print ('######################  save_weights  ######################')
    model.save_weights(save_path+'/model.hdf5')
'''

if __name__ == '__main__':
    samples = []
    for line in range(1000):
        samples.append(['../Train/CameraRGB/%d.png' % line, '../Train/CameraSeg/%d.png' % line])

    train_samples, validation_samples = train_test_split(samples, test_size=0.15)
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=36)
    validation_generator = generator(validation_samples, batch_size=36)

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

'''
if __name__ == '__main__2':
    print ('******************************************************************')
    model_name = 'AtrousFCN_Resnet50_16s'
    #model_name = 'Atrous_DenseNet'
    #model_name = 'DenseNet_FCN'
    batch_size = 16
    batchnorm_momentum = 0.95
    epochs = 250
    lr_base = 0.01 * (float(batch_size) / 16)
    lr_power = 0.9
    resume_training = False
    if model_name is 'AtrousFCN_Resnet50_16s':
        weight_decay = 0.0001/2
    else:
        weight_decay = 1e-4
    target_size = (320, 320)
    
    dataset = 'LYFT'
    if dataset == 'LYFT':
        # pascal voc + berkeley semantic contours annotations
        train_file_path = os.path.expanduser('../combined_imageset_train.txt')
        val_file_path   = os.path.expanduser('../combined_imageset_val.txt')
        data_dir        = os.path.expanduser('../Train/CameraRGB')
        label_dir       = os.path.expanduser('../Train/CameraSeg')
        data_suffix='.png'
        label_suffix='.png'
        classes = 3
        loss_shape = (target_size[0] * target_size[1] * classes,)
    if dataset == 'VOC2012_BERKELEY':
        # pascal voc + berkeley semantic contours annotations
        train_file_path = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        # train_file_path = os.path.expanduser('~/.keras/datasets/oneimage/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        val_file_path   = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_val.txt')
        data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
        label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/combined_annotations')
        data_suffix='.jpg'
        label_suffix='.png'
        classes = 21
    if dataset == 'COCO':
        # ###################### loss function & metric ########################
        train_file_path = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        # train_file_path = os.path.expanduser('~/.keras/datasets/oneimage/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        val_file_path   = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')
        data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
        label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/SegmentationClass')
        loss_fn = binary_crossentropy_with_logits
        metrics = [binary_accuracy]
        loss_shape = (target_size[0] * target_size[1] * classes,)
        label_suffix = '.npy'
        data_suffix='.jpg'
        ignore_label = None
        label_cval = 0


    # ###################### loss function & metric ########################
    if dataset == 'VOC2012' or dataset == 'VOC2012_BERKELEY':
        loss_fn = softmax_sparse_crossentropy_ignoring_last_label
        metrics = [sparse_accuracy_ignoring_last_label]
        loss_shape = None
        ignore_label = 255
        label_cval = 255

    # Class weight is not yet supported for 3+ dimensional targets
    # class_weight = {i: 1 for i in range(classes)}
    # # The background class is much more common than all
    # # others, so give it less weight!
    # class_weight[0] = 0.1
    class_weight = None

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)
    print ('222222222222222222222222222222222222222222222222222222222222')
    train(batch_size, epochs, lr_base, lr_power, weight_decay, classes, model_name, train_file_path, val_file_path,
          data_dir, label_dir, target_size=target_size, batchnorm_momentum=batchnorm_momentum, resume_training=resume_training,
          class_weight=class_weight, loss_fn=binary_crossentropy_with_logits, metrics=[binary_accuracy], loss_shape=loss_shape, data_suffix=data_suffix,
          label_suffix=label_suffix, ignore_label=None, label_cval=None)
'''