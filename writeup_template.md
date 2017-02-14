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

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of five convolutional layers, each followed by an activation ELU layer.  These are used to extract features from the images.  The first four convolutional filters are 5x5 and the last one is 3x3.  The layers increase in depth, starting at 3 and ending at 64.
These are followed by a flatten layer and four fully connected (dense) layers.  The purpose of these is to train the model to map the extracted features to outputs.  

####2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting.  The first one happens after _________ and the second one happens before the last fully connected layer.

CHANGE THIS:
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer and mse (mean squared error).

####4. Appropriate training data

Training data used was provided by Udacity.  It included three camera angles - center, left and right.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to pick an existing known model and follow its patterns, with some changes to make it work for my project.

My first step was to use a convolutional neural network model similar to the Nvidia pipeline.   I thought this model might be appropriate because it had a good number of convolutional layers, which I thought would give the model enough opportunities to extract important features.

CHANG THIS:
To combat the overfitting, I added a couple dropout layers to the model.

Then I ...

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

Then I realized another problem could be that since the behicle mostly drives straight, it can be biased to drive straight most of the time.  This matched my observations in the simulator, since the vehicle was reluctant to turn and would only turn sharply when it was close to the side of the road.
To fix this, I removed the data with steering angle of 0 from the training set.  I set a threshold to be 0.001 to account for rounding errors.

This got me a point where I could drive the entire loot twice (at least, probably more) without crashing.
At this point, the model is still not perfect and swerves a lot, occasionally hitting the sides of the road.  It doesn't handle bridges very well and sticks to the side of the bridge for a long time.
However, it has recovered successfully every time so far. 


####2. Final Model Architecture

The final model architecture consists of a convolution neural network with the following layers and layer sizes ...

OPTIONAL:
Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

I tried the drive the track on my own at first, but didn't do a very good job and would end up hitting the sides occasionally.  I have read in discussions that people had a hard time driving it with a keyboard because the steering angles end up beeing too extreme.
I decided to go with the Udacity provided data for now, assuming that if I need more data, I will try driving with a controller.

I took the Udacity provided images and converted them from BGR to RGB.  I cropped them to include only the important parts that show the curvature of the road.  I then flipped all of the center images and added them to my training set.
Then I took the left and right camera images and added them to the data set with properly adjusted steering angles.  I ended up adjusting them by .13.
I normalized the image data as a preprocessing step, so the would range between -1 and 1.

CHANGE THIS:

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
