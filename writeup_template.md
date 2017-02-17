#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py does image preprocessing, builds and trains the model, saves in into a separate file
* drive.py for driving the car in autonomous mode
* model.h5 contains a trained model 
* writeup_report.md summarizes the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the steps of the implementation of the model, in order.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of five convolutional layers, each followed by an activation ELU layer.  These are used to extract features from the images.  The first four convolutional filters are 5x5 and the last one is 3x3.  The layers increase in depth, starting at 12 and ending at 64.
These are followed by a flatten layer and four fully connected (dense) layers.  The purpose of these is to train the model to map the extracted features to outputs.  

####2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting.  The first one happens after ______the Flatten layer and the second one happens before the last fully connected layer.
The data used for training the model is shuffled every time before training so it does not have a dependency on the order of the input.

####3. Model parameter tuning

The model used an adam optimizer and mse (mean squared error), which means the learning rate was not set manually.

####4. Appropriate training data

Training data used was provided by Udacity.  It included three camera angles - center, left and right.
I've augmented the training data by adding some of my own.  I added some data for just general driving around the track and some extra data for particularly difficult areas, like sharp turns.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to pick an existing known model and follow its patterns, with some changes to make it work for my project.

My first step was to use a convolutional neural network model similar to the Nvidia pipeline.   I thought this model might be appropriate because it had a good number of convolutional layers, which I thought would give the model enough opportunities to extract important features.

Then I added a couple of droupout layers to the model in order to prevent overfitting.

After my first semi-successful training of a model, I tried to run it in the simulator.  The vehicle turned left and into the lake.  

I added more proprocessing steps to fix the problem.  I figured the excessive left turning was caused by my data having a lot of left turns, which ended up training the model to favor left turns.  
To balance this out, I flipped the images and added a flipped version of each image to my training set.
This helped the car steer more or less straight.  However, it still couldn't make it past the first turn.

My next step was to normalize the input to -1 to 1 range.  I've been meaning to do this for a while, since normalizing considered good practice.  The main purpose of it was to help the model take advantage of the nonlinearity that the ELU layer added.  If the values in the data ranged between 0 and 255, it would be difficult to take advantage of the ELU to represent a nonlinear function.
This improved the results in the simulator significantly.  The vehicle was able to make turns and got past the first bridge.
However, it still went off the road after the bridge.

My next step was to add recovery data, since the vehicle didn't seem to know how to recover once it ended up too close to the side of the road.
I used the left and right camera images from the Udacity data and added an offset to the steering angles and added all of that to my training set.
This showed a slight improvement, but the vehicle would still crash shortly after the first bridge.

Then I realized another problem could be that since the vehicle mostly drives straight, it can be biased to drive straight most of the time.  This matched my observations in the simulator, since the vehicle was reluctant to turn and would only turn sharply when it was close to the side of the road.
To fix this, I removed the data with steering angle of 0 from the training set.  I set a threshold to be 0.001 to account for rounding errors.

This got me a point where I could drive the entire loop twice (at least, probably more) without crashing.
At this point, the model was still not perfect and swerved a lot, occasionally hitting the sides of the road.  It didn't handle bridges very well and stuck to the side of the bridge for a long time.
However, it has recovered successfully every time.

In order to improve that and avoid driving up on the edges of the road, I recorded some of my own data so I would have more data to train the model on.
Additional training has improved the model's ability to drive over the bridge and to handle turns.  I also experimented with different thresholds for removing zero steering values and correction angles using side cameras.
I noticed the model did slightly better with a larger number of epochs (around 7 seemed to be a good number for me).
This got the vehicle to drive around the track without getting stuck on the bridge or going off on the sides.  However, the car still looks drunk and swerves a lot. 


####2. Final Model Architecture

The final model architecture consists of a convolution neural network with the following layers and layer sizes:

Input
Convolutional layer: 12  5x5 filters, stride of 4
ELU Activation
Convolutional layer: 24 5x5 filters, stride of 2
ELU Activation
Convolutional layer: 36 5x5 filters, stride of 2
ELU Activation
Convolutional layer: 48 5x5 filters, stride of 2
ELU Activation
Max Pooling: pool size 2x2
Convolutional layer: 64 3x3 filters, stride of 2
ELU Activation
Flatten
Dropout: drops 20% of input units
ELU Activation
Fully Connected
ELU Activation
Fully Connected
ELU Activation
Fully Connected
Dropout: drops 50% of input units
ELU Activation
Fully Connected


####3. Creation of the Training Set & Training Process

I tried to drive the track on my own at first, but didn't do a very good job and would end up hitting the sides occasionally.  I have read in discussions that people had a hard time driving it with a keyboard because the steering angles end up beeing too extreme.
I decided to go with the Udacity provided data for now, assuming that if I need more data, I will try driving with a controller.

I took the Udacity provided images and converted them from BGR to RGB.  I cropped them to include only the important parts that show the curvature of the road.  I then flipped all of the center images and added them to my training set.
Then I took the left and right camera images and added them to the data set with properly adjusted steering angles.  I ended up adjusting them by .13.
I normalized the image data as a preprocessing step, so the would range between -1 and 1.

After the collection process, I had 25912 of data points. By the time I removed zero steering angles, I had 20659 data points (this number varies since I use a random number to determine how many zero values to keep).

The data is randomly shuffled and 20% of is us used for validation, which in this case came out to 5165.

