# Semantic Segmentation on CARLA dataset
## Introduction
This project is for a image semantic segmantation challenge by Lyft & Udacity, the algorithm was benchmarked on a video recorded using [CARLA](http://carla.org/), CARLA is an open-source simulator for autonomous driving research.

This is a postmortem for this project. This article will discuss the 3 following topics:

1. How I setup the architecture of the project,
1. What are the challenges and what's my solution to them,
1. What are the potential improvement for this project.

Now, let's get started!

## Architecture of the project

### Fully Convolutional Network

I used Fully Convolutional Network in this project. And I selected ResNet. I used the [Keras-FCN](https://github.com/aurora95/Keras-FCN) framework to start with.

The idea of fully convolutional network is to use a neural network wihtout fully-connected layers, which means all layers are convolutional layer. At the end, you add a convolution layer to represent the semantic segmentation result.

I used Keras in this project, with tensorflow backend. Images are scaled to 320x320 before fitting into the network, and the semantic segmentation result will be scale back to 800x600.

**Before I get into more detal of FCN, here are some other solutions that I came up with.**

### *Mask R-CNN

[Mask R-CNN](https://arxiv.org/abs/1703.06870) is an object instance segmentation deep learning framework base on faster R-CNN. Mask R-CNN detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. In other words, **faster R-CNN create bounding box for each instance in the image, while mask R-CNN added a semantic segmentation stage within each bounding box.**

Here are some reasons I didn't select R-CNN for this task.
1. We don't have bounding box in training dataset, which means if we draw the bounding box according to semantic segmentation result, it won't be able to seperate vehicles that are overlay.
1. Spatial information are lost during the detection process, a vehicle on the left lane and in front of us will have different shape, but in the semantic segmentation stage of the mask R-CNN, this information has already disappear.
1. Using mask R-CNN on road surface doesn't seem to be a good fit in this situation to me. I'll prefer draw the bounding box by my self because the detection result of road surface would be very consistent. Which will give us just another fully convolutional network

### *Histogram of Oriented Gradient (HOG) + Linear SVM + Semantic Segmentation
This is another solution that I came up with, I didn't choosed this solution.

In my [Vehicle Detection project](https://github.com/EvanWY/CarND-Vehicle-Detection/blob/master/writeup.md), I used Histogram of Oriented Gradients (HOG) with linear SVM to classify vehicle, and using sliding window search to implemented vehicle detection framework. I optimized the framework using heatmap across multiple frames. One of the solution is to add a semantic segmentation stage on top of the detection bounding box result.

**Pros:** By applying multiple stages (classification, detection, heatmap, segmentation), this architecture would give me more control on each stage of the segmentation process. It would be easier to identify which step is the bottleneck.

**Cons:**
1. When 2 vehicle merged together in the image, It will be difficut to draw 2 bounding box for each of them to train the detector (bounding box not provided in the test dataset, need to draw them according to sem-seg result);
1. The sliding window + detection solution would not be fast enough to run in realtime;
1. Need to come up with another solution for road surface segmentation (Could be extended from this lane [dection](https://github.com/EvanWY/CarND-LaneLines-P1/blob/master/writeup.md) project)
1. **What's more important:** Why not use deep learning? It performed much better than conventional computer vision algorithm in tasks like this. :)

## Challenges

### Data Augmentation & Recording

The original test dataset only have 1000 training images, which is not sufficient for training. If we have a system that can train at 5 fps (roughly divide "running at 10" fps by 2), it would only take 200 seconds for the training to finish an epoch.

I [randomly crop both raw image and semantic segmentation image](https://github.com/EvanWY/CARLASemSeg/blob/3bf3dc6bcc5c936e81f6f2a63481bc0125746ccd/Keras-FCN/train.py#L43-L49) at the beginning of training for every images. I also [flip the images](https://github.com/EvanWY/CARLASemSeg/blob/3bf3dc6bcc5c936e81f6f2a63481bc0125746ccd/Keras-FCN/train.py#L62-L63) to create a new training image.

Beside data augmentation, I also expand the dataset by recording new data using CARLA. Here is the [script](https://github.com/EvanWY/CARLASemSeg/blob/master/Keras-FCN/record_data.py) I used to connect to CARLA simulator and save images.

I [upload saved image to AWS S3](https://github.com/EvanWY/CARLASemSeg/blob/master/Keras-FCN/record_data.py#L100) for future uses.

### "Live" Training - connect Trainer directly with CARLA simulator

To make the system even more scalable, I setup a client-server training model, where CARLA simulator is a server and Keras trainning script is a client. Here is the "live" trining [script](https://github.com/EvanWY/CARLASemSeg/blob/master/Keras-FCN/live_train.py).

In this system, the trining script and CARLA run simultaneously. Training script asks CARLA to proceed to next frame, and CARLA render next frame and return both raw image and segmentation image to the training script. Training script will train the model during this process and ask CARLA to start another episode after several frames.

However, this system is not as fast as I expected. I found that the bottleneck was the CARLA rendering and starting new episode. I have to increase the amount of data augmentation.

There is another drawback in this system, training images will fit into the model sequentially, not randomly, which will lead to overfitting.

**Things I would like to try:**

**With the client-server model, it's easy to expand to a multi-simulator system**, which means I can have a fleet of render server running simulation and connect one trainning machine to multiple simulation machine. By doing this, **we can also tackle the challenge where training image are sequential**, because multiple simulator will run in different scene.

### loss function & Hyper Parameters Tuning

The scoring algorithm in this project seperated percision and recall, where vehicle score will more affected by recall, while road surface (drivable surface) will affected by percision. Which actually make sense because we don't want to indentify a car pixel as non-car and we don't want to identify a obstacle as drivable surface.

Because of this, I decided to treat each pixel as a **multi class classification** problem instead of a **single class classification** problem, which means a pixel can be one of the 4 states:

1. None
1. Car
1. Road
1. Both Car and Road

That sounds counter intuitive right? But because of the nature of our scoring system, it actually make sense to do so. Here is an example:

Let's assume that the network is classifying pixel A, it decided A has 50% chance to be "Road", and 50% chance to be "Car", and 0% chance to be "None". Now we are filling the segmentation mask, it might give us more expectation in scoring if we set "Road" and "Car" to both true. (Paramters selected in our scoring system made it not likely to happen, but this implementation still make it easier to tune the result)

Eventually, that give me 2 parameters to tune: the threshold for both "Car" and "Road"

I used [this script](https://github.com/EvanWY/CARLASemSeg/blob/master/Keras-FCN/hyperoptim.py) to tune the two hyperparameters. Because the two hyperparameters are independent from each other, I can tune them seperately to get the best result for both Car and Road.

### Temporal Infomation

### FPS optimization: Tensorflow stating

## Things I would like to try

crop image
add time info
fps optimize: pack testing data
live training with multiple server
