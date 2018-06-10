# Semantic Segmentation on CARLA dataset
## Introduction
This project is for a image semantic segmantation challenge by Lyft & Udacity, the algorithm was benchmarked on a video recorded using [CARLA](http://carla.org/), CARLA is an open-source simulator for autonomous driving research.

This is a postmortem for this project. This article will discuss the 3 following topics:

1. How I setup the architecture of the project,
1. What are the challenges and what's my solution to them,
1. What are the potential improvement for this project.

Now, let's get started!

## Deep Neural Network Architecture of the project
### Existing solutions
#### 1. Histogram of Oriented Gradient (HOG) + Linear SVM + Semantic Segmentation
In my [Vehicle Detection project](https://github.com/EvanWY/CarND-Vehicle-Detection/blob/master/writeup.md), I used Histogram of Oriented Gradients (HOG) with linear SVM to classify vehicle, and using sliding window search to implemented vehicle detection framework. I optimized the framework using heatmap across multiple frames.

One of the solution is to add a semantic segmentation stage on top of the detection bounding box result.

***Pros and Cons:***
Pros: By applying multiple stages (classification, detection, heatmap, segmentation), this architecture would give me more control on each stage of the segmentation process. It would be easier to identify which step is the bottleneck.
Cons: 

