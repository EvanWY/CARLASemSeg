import cv2
import numpy as np
import sys, os
from subprocess import call

# img = cv2.imread('/Users/yang/Desktop/img0001790.png')
# img2 = cv2.imread('/Users/yang/Desktop/seg0001790.png')

# cv2.imshow('image1', img)
# cv2.imshow('image2', img2 * 20)


# cv2.waitKey(0)
# cv2.destroyAllWindows()

log = """/home/workspace/____/graderh: !

########################################
############ Starting Grader ###########
########################################


Evaluating submission...
Using TensorFlow backend.
2018-06-03 07:02:07.267091: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-06-03 07:02:07.267136: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-06-03 07:02:07.267148: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-06-03 07:02:07.267155: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2018-06-03 07:02:07.267171: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2018-06-03 07:02:07.354600: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-06-03 07:02:07.355049: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties:
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:04.0
Total memory: 11.17GiB
Free memory: 11.09GiB
2018-06-03 07:02:07.355098: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0
2018-06-03 07:02:07.355123: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y
2018-06-03 07:02:07.355152: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
2018-06-03 07:02:22.379514: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)

Your program has run, now scoring...

Your program runs at 9.174 FPS

Car F score: 0.773 | Car Precision: 0.675 | Car Recall: 0.802 | Road F score: 0.974 | Road Precision: 0.990 | Road Recall: 0.916 | Averaged F score: 0.874

If you are satisfied with your results and have no errors, make sure you have filled out your JWT file and then submit your solution by typing 'submit' at the command line.

########################################
############ Ending Grader #############
########################################"""


