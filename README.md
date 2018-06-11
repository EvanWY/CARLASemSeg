# Semantic Segmentation on CARLA dataset
## Introduction
This project is for an image semantic segmentation challenge by Lyft & Udacity, the algorithm was benchmarked on a video recorded using [CARLA](http://carla.org/), CARLA is an open-source simulator for autonomous driving research.

This is a postmortem for this project. This article will discuss the 3 following topics:

1. How I set up the architecture of the project.
1. What are the challenges and what's my solution to them?
1. What are the potential improvements for this project?

Now, let's get started!

## Architecture of the project

### Fully Convolutional Network

I used Fully Convolutional Network in this project. And I selected ResNet. I used the [Keras-FCN](https://github.com/aurora95/Keras-FCN) framework to start with.

The idea of the fully convolutional network is to use a neural network without fully-connected layers, which means all layers are convolution layer. In the end, you add a convolution layer to represent the semantic segmentation result.

I used Keras in this project, with Tensorflow backend. Images are scaled to 320x320 before fitting into the network, and the semantic segmentation result will be scale back to 800x600.

### Other Possible Frameworks

Before I get into more detail of FCN, here are some other solutions that I came up with.

| Name           | Description  |
| ------------- |-------------|
|  Mask R-CNN | [Mask R-CNN](https://arxiv.org/abs/1703.06870) is an object instance segmentation deep learning framework base on faster R-CNN. Mask R-CNN detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. In other words, **faster R-CNN create a bounding box for each instance in the image, while mask R-CNN added a semantic segmentation stage within each bounding box.**  Here are some reasons I didn't select R-CNN for this task.<br /><br /> 1. We don't have bounding boxes in training dataset, which means if we draw the bounding box according to semantic segmentation result, it won't be able to separate vehicles that are overlayed.<br /><br /> 2. Spatial information is lost during the detection process, a vehicle on the left lane and in front of us will have a different shape, but in the semantic segmentation stage of the mask R-CNN, this information has already disappeared.<br /><br /> 3. Using mask R-CNN on road surface doesn't seem to be a good fit in this situation to me. I'll prefer to draw the bounding box by my self because the detection result of road surface would be very consistent. Which will give us just another fully convolutional network<br /><br />|
| Histogram of Oriented Gradient (HOG) + Linear SVM + Semantic Segmentation |This is another solution that I came up with, I didn't chose this solution.<br />In my [Vehicle Detection project](https://github.com/EvanWY/CarND-Vehicle-Detection/blob/master/writeup.md), I used Histogram of Oriented Gradients (HOG) with linear SVM to classify vehicle and using sliding window search to implemented vehicle detection framework. I optimized the framework using heatmap across multiple frames. One of the solutions is to add a semantic segmentation stage on top of the detection bounding box result.<br /><br />**Pros:** <br />By applying multiple stages (classification, detection, heatmap, segmentation), this architecture would give me more control on each stage of the segmentation process. It would be easier to identify which step is the bottleneck.<br /><br />**Cons:** <br />1. When 2 vehicles merged together in the image, It will be difficult to draw 2 bounding box for each of them to train the detector (bounding box not provided in the test dataset, need to draw them according to sem-seg result);<br /><br />2. The sliding window + detection solution would not be fast enough to run in real-time;<br /><br />3. Need to come up with another solution for road surface segmentation (Could be extended from this lane [dection](https://github.com/EvanWY/CarND-LaneLines-P1/blob/master/writeup.md) project)<br /><br />4. **What's more important:** Why not use deep learning? It performed much better than conventional computer vision algorithm in tasks like this. :)|

## Challenges

### Data Augmentation & Recording

The original test dataset only has 1000 training images, which is not sufficient for training. If we have a system that can train at 5 fps (roughly divide "running at 10" fps by 2), it would only take 200 seconds for the training to finish an epoch.

I [randomly crop both raw image and semantic segmentation image](https://github.com/EvanWY/CARLASemSeg/blob/3bf3dc6bcc5c936e81f6f2a63481bc0125746ccd/Keras-FCN/train.py#L43-L49) at the beginning of training for every image. I also [flip the images](https://github.com/EvanWY/CARLASemSeg/blob/3bf3dc6bcc5c936e81f6f2a63481bc0125746ccd/Keras-FCN/train.py#L62-L63) to create a new training image.

Beside data augmentation, I also expand the dataset by recording new data using CARLA. Here is the [script](https://github.com/EvanWY/CARLASemSeg/blob/master/Keras-FCN/record_data.py) I used to connect to CARLA simulator and save images.

I [upload saved images to AWS S3](https://github.com/EvanWY/CARLASemSeg/blob/master/Keras-FCN/record_data.py#L100) for future uses.

### "Live" Training - connect Trainer directly with CARLA simulator

To make the system even more scalable, I set up a client-server training model, where CARLA simulator is a server and Keras training script is a client. Here is the "live" training [script](https://github.com/EvanWY/CARLASemSeg/blob/master/Keras-FCN/live_train.py).

In this system, the training script and CARLA run simultaneously. Training script asks CARLA to proceed to next frame, and CARLA renders next frame and returns both raw image and segmentation image to the training script. Training script will train the model during this process and ask CARLA to start another episode after several frames.

However, this system is not as fast as I expected. I found that the bottleneck was the CARLA rendering and starting a new episode. I have to increase the amount of data augmentation.

There is another drawback in this system, training images will fit into the model sequentially, not randomly, which will lead to overfitting.

**Things I would like to try:**

**With the client-server model, it's easy to expand to a multi-simulator system**, which means I can have a fleet of render server running the simulation and connect one training machine to multiple simulation machines. By doing this, **we can also tackle the challenge where training image is sequential** because multiple simulators will run in different scenes.

### loss function & Hyper Parameters Tuning

The scoring algorithm in this project separate precision and recall, where vehicle score will more be affected by the recall, while road surface (drivable surface) will be affected by precision. Which actually make sense because we don't want to identify a car pixel as non-car and we don't want to identify an obstacle as the drivable surface.

Because of this, I decided to treat each pixel as a **multi-class classification** problem instead of a **single class classification** problem, which means a pixel can be one of the 4 states:

1. None
1. Car
1. Road
1. Both Car and Road

That sounds counter-intuitive, right? But because of the nature of our scoring system, it actually makes sense to do so. Here is an example:

Let's assume that the network is classifying pixel A, it decided A has 50% chance to be "Road", and 50% chance to be "Car", and 0% chance to be "None". Now we are filling the segmentation mask, it might give us more expectation in scoring if we set "Road" and "Car" to both true. (Parameters selected in our scoring system made it not likely to happen, but this implementation still make it easier to tune the result)

Eventually, that give me 2 parameters to tune: the threshold for both "Car" and "Road"

I used [this script](https://github.com/EvanWY/CARLASemSeg/blob/master/Keras-FCN/hyperoptim.py) to tune the two hyperparameters. Because the two hyperparameters are independent of each other, I can tune them separately to get the best result for both Car and Road.

### FPS optimization: Tensorflow stating

I noticed that the actual validation process can run above 10 FPS, while Tensorflow initialization takes up a lot of time which decrease the FPS to lower than 10 FPS. To optimize the FPS, I use the client-server architecture, using a [backend](https://github.com/EvanWY/CARLASemSeg/blob/master/Keras-FCN/inference_server.py) running Keras, which is already initialized and waiting for the [client](https://github.com/EvanWY/CARLASemSeg/blob/master/Keras-FCN/inference_client.py) to connect. 

### Temporal Information

After finishing all optimization/tuning above, I started to train the model for the last time, and validation loss stays at around 0.0073. At this point, I am thinking about what else I can do to improve the system even more.

This time, I stop looking at details, I view the whole system as a **black box**, and I realize that if I want to improve the system even more, I'll need to input more information to the system. (preprocessing such as converting colorspace doesn't add information, it just preprocess the data to make it easier to learn, which can eventually learn by the neural network)

I decided that temporal information is the new information I'll add to the system. I came up with 3 solutions:
1. Recurrent Neural Network
1. Stacking multiple frames as input to FCN.
1. Add a heat map to the result, which preserves segmentation information for next frame.

At this point, I only have couple hours to deadline, I don't have time to train a new network, so I decided to do the heatmap trick.

The basic idea is to mix the segmentation from the last frame with the current frame, with a parameter "Fade". The new pixel value would be *Fade * oldPixel + (1-Fade) * newPixel*

## Things I would like to try
Here are something I would like to try to improve my system:

### Multi Render Server "Live" Training
As mentioned above, I believe having multiple servers running CARLA would make the live training much better.

### Crop Image
Initially, I have the idea of **transfer learning**, which is to use existing model and it's weight. Which force me to fit my image size with the pre-trained model (320x320). In the end, I train the entire model from scratch, which means I probably could have better tuned the model layer sizes according to the images.

### Time Information
As mentioned above, I believe both recurrent neural network and stacking multiple frames as input to FCN could improve the system a lot by providing temporal information.

### Fps optimize: pack testing data
When running on scoring, I'm using frame by frame validation. I believe the FPS will improve if I pack multiple frames together before sending into the model.
