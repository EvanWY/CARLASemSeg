from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import pickle

import time
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
from PIL import Image as PImage
from subprocess import call

def sim_frame_generator():
    call(['aws', 's3', 'sync', '--quiet', '/home/workspace/CARLASemSeg/Train', 's3://yang-carla-train'])
    frame = 100000
    last_frame_time = time.time()
    print ('initializing CARLA client connection')
    with make_carla_client('localhost', 2000, timeout=300) as client:
        try:
            print('CarlaClient connected !')
            while 1:
                #init
                settings = CarlaSettings()
                settings.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=random.choice([30, 50, 120, 200]),
                    NumberOfPedestrians=random.choice([0, 10, 20, 30]),
                    WeatherId=random.choice([1, 2, 8, 1, 2, 8, 1, 2, 3, 6, 7, 8]),
                    QualityLevel='Epic')
                settings.randomize_seeds()
                #settings.randomize_weather()

                camera0 = Camera('CameraRGB')
                camera0.set_image_size(800, 600)
                camera0.set_position(1.3, 0, 1.3)
                #camera0.FOV = 60
                settings.add_sensor(camera0)

                camera1 = Camera('CameraSemSeg', PostProcessing='SemanticSegmentation')
                camera1.set_image_size(800, 600)
                camera1.set_position(1.3, 0, 1.3)
                #camera1.FOV = 60
                settings.add_sensor(camera1)

                scene = client.load_settings(settings)

                number_of_player_starts = len(scene.player_start_spots)
                player_start = random.randint(0, max(0, number_of_player_starts - 1))

                client.start_episode(player_start)
                
                for xx in range(500):
                    measurements, sensor_data = client.read_data()
                    for name, measurement in sensor_data.items():
                        image = PImage.frombytes(
                            mode='RGBA',
                            size=(measurement.width, measurement.height),
                            data=measurement.raw_data,
                            decoder_name='raw')
                        color = image.split()
                        image = PImage.merge("RGB", color[2::-1])
                        
                        if name == 'CameraRGB':
                            img = image
                        elif name == 'CameraSemSeg':
                            seg = image
                    
                    img.save('/home/workspace/CARLASemSeg/Train/CameraRGB/%07d.png'%frame,"PNG")
                    seg.save('/home/workspace/CARLASemSeg/Train/CameraSeg/%07d.png'%frame,"PNG")
                    frame += 1
                    if (frame >= 200000):
                        return
                    if (frame % 100 == 0):
                        print()
                        print ("saving frame id: {}, time:{}".format(frame, time.time()))
                        print()

                    fps = 1.0/(time.time() - last_frame_time)
                    last_frame_time = time.time()
                    sys.stdout.write("\r" + str(fps))
                    sys.stdout.flush()

                    control = measurements.player_measurements.autopilot_control
                    control.steer += random.uniform(-0.1, 0.1)
                    client.send_control(control)

                call(['aws', 's3', 'sync', '--quiet', '/home/workspace/CARLASemSeg/Train', 's3://yang-carla-train'])
                    
        finally:
            pass
            

sim_frame_generator()
    
