# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
---

[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./center.jpg "Center Position"
[image3]: ./center2.jpg "Center Position"
[image4]: ./flip.jpg "Flipped Image"


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_all_4conv_RGB.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_all_4conv_RGB.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 120 (model.py lines 83-100) 

The model includes RELU layers to introduce nonlinearity (code lines 87,89,91,93), and the data is normalized in the model using a Keras lambda layer (code line 85). 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py line 97). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 102).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and opposite direction driving. 


---

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet5. The results were OK for the first try but there was a lot of impovement to be done. I increased the data set, and the results became better and better up to a certain point. Then I started looking for other network architectures and I went with NVIDIA CNN model for autonomous cars. Then structure is similar but my model contains only 4 convolutional layers and the original one 5. Also I used different image shapes.

To combat the overfitting, I modified the model and I added a dropout layer.Then I used different data sets to be sure the model was not overfitting.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 83-100) consisted of a convolution neural network with the following layers and layer sizes:
- Normalization layer 3@65X320
- Convolutional layer 24@30X158
- Convolutional layer 36@13X77
- Convolutional layer 48@4X36
- Convolutional layer 64@1X17
- Fully Connected layer 1088
- Fully Connected layer 120
- Dropout layer 15%
- Fully Connected layer 50
- Fully Connected layer 10

Here is a visualization of the architecture

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a few laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to drive in certain conditions.

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help because the first track has only left corners. For example, here is an image that has then been flipped:

![alt text][image3]
![alt text][image4]


After the collection process, I had 90.000 number of data points. I then preprocessed this data by cropping the top and the bottom of the image. Also while training all the data has been converted from BGR to RGB. I haven't realised this until, but cv2 is reading the image with BGR and the test with RGB

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 because mse is not decreasing that much with more epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.

# The vehicle is able to drive autonomously around both tracks without leaving the road. See videos track1 and track2
