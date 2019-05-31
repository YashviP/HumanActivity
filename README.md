# HumanActivity

Action Recognition refers to an algorithm that the computer system uses to automatically recognize what human action is being or was performed , in a given video sequence .  Actually it  is problem of  classifying action and assigning it into a label .

# Dataset

I used this dataset, the link is given below
http://www.nada.kth.se/cvap/actions/

I made this project using opencv-python for video processing and feature extraction  and support vector machines  for classification.

# System Description

#  Moving object Detection

Background is redundant part in recognizing action because action is related  only with the body movements of an object. So,moving object is segmented from background.

# Feature Extraction

1. Average of HOG feature 

A feature descriptor is a representation of an image or an image patch that simplifies the image by extracting useful information and throwing away extraneous information.

In the HOG feature descriptor, the distribution ( histograms ) of directions of gradients ( oriented gradients ) are used as features. Gradients ( x and y derivatives ) of an image are useful because the magnitude of gradients is large around edges and corners ( regions of abrupt intensity changes ) and we know that edges and corners pack in a lot more information about object shape than flat regions.


2. Displacement of Object Centroid 

Displacement is measure of change in position of an object . It is the measure how far the object has moved from its original position.
we have estimated displacement using centroid of object peripheral.

3. velocity of object

Rate of change of displacement . velocity is speed of object and its direction of motion.

4.  Local Binary Pattern 

Local Binary Patterns, or LBPs for short, are a texture descriptor .LBPs compute a local representation of texture. This local representation is constructed by comparing each pixel with its surrounding neighborhood of pixels.


# Model training 

I created feature matrix , and used this for my model training .
I used Support Vector Machines.

# testing 

Now, predict the label for action by passing action video stream.

