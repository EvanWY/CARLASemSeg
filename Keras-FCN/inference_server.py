import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
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
import zmq

# Define encoder function
def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

print ("start loading model")
model = zerg_model(batch_shape=[1, 320, 320, 3])
model.load_weights('terran_model.h5')
print ("finish loading model")


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    print ("waiting for message")
    road_th = float(socket.recv_string())
    veh_th = float(socket.recv_string())
    road_fade = float(socket.recv_string())
    veh_fade = float(socket.recv_string())
    message = socket.recv_string()
    
    print ("got message! road_th={}, veh_th={}, road_fade={}, veh_fade={}, message={}".format(road_th,veh_th,road_fade,veh_fade,message))
    file = message

    print ("start reading video")
    video = skvideo.io.vread(file)
    print ("done reading video")

    answer_key = {}

    # Frame numbering starts at 1
    frame = 1


    print ("start processing frames ...")
    seg = np.zeros(320,320,2)
    for rgb_frame in video:
        
        img = cv2.resize(rgb_frame, (320, 320), interpolation = cv2.INTER_CUBIC)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        new_seg = model.predict(img.reshape(1,320,320,3)).reshape(320,320,2)
        if frame % 200 == 2:
            print ('processing frame:' + str(frame))
            seg = new_seg
        else:
            seg[:,:,0] *= road_fade
            seg[:,:,1] *= veh_fade
            seg += new_seg

        seg_road = (seg[:,:,0] > road_th).astype(np.uint8)
        seg_vehicle = (seg[:,:,1] > veh_th).astype(np.uint8)
        
        seg_road_fullsize = cv2.resize(seg_road, (800, 600), interpolation = cv2.INTER_NEAREST)
        seg_vehicle_fullsize = cv2.resize(seg_vehicle, (800, 600), interpolation = cv2.INTER_NEAREST)

        answer_key[frame] = [encode(seg_vehicle_fullsize), encode(seg_road_fullsize)]
        
        # Increment frame
        frame+=1
    print ("done processing frames")

    # Print output in proper json format
    #print (json.dumps(answer_key))
    socket.send_string(json.dumps(answer_key))