#**Behavioral Cloning**

The goals  of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture.png "Model Architecture"
[image2]: ./examples/crop_resize.png "Crop - Resize"
[image3]: ./examples/udacity.png "Angle distribution"
[image4]: ./examples/udacity_left_right.png "Angle distribution"
[image5]: ./examples/flip.png "Flip Image"
[image6]: ./examples/udacity_augmentation.png "Angle distribution"
[image7]: ./examples/mydata_angle.png "Angle distribution"
[image8]: ./examples/mydata_augmented_angle.png "Angle distribution"


---
### Files Submitted & Code Quality

#### 1. Submission files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* data_processor.py contains the script used for preprocessing
* project_output.mp4 simulator output in autonomous mode

#### 2. Setup
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```  
Creating and training a new model
```sh
python main.py <path_to_data_folder>
```

Transfer Learning
```sh
python main.py <path_to_data_folder> --load <model.h5 file>
```


#### 3. Submission code

The model.py file contains the code for training and saving the convolution neural network. It also contains a code to load and save the model.

data_processing.py contains the data pre-processing and data augmentation code


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I'm using the model described in the paper from  [NVIDIA](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)    

![alt text][image1]

The network consists of 9 layers, including a normalization layer, 5 convolutional layers
and 3 fully connected layers. The input image is split into YUV planes and passed to the network.
The first layer of the network performs image normalization. The normalizer is hard-coded and is not
adjusted in the learning process. Performing normalization in the network allows the normalization
scheme to be altered with the network architecture and to be accelerated via GPU processing.  Strided convolutions are used in the
first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution
with a 3×3 kernel size in the last two convolutional layers

#### 2. Attempts to reduce overfitting in the model
I added dropout layer with different droput percentage after each of the fully connected layers in order to reduce overfitting. See the create_model() function in model.py file

Model was trained and validated with Udacity provided data and also data collected by me. I used transfer learning to continuously improve the model

####3. Model parameter tuning

The model uses an Adam optimizer to minimize the means squared error between it's output and the recorded steering angle from the simulator that corresponds to the image of the road at that time.. I used a very low learning rate of 0.00001 while tuning the model with transfer learning

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I started with the Udacity provided training data and then improved the model by training on the data collected by me. Overall I used more than 20,000 data points including around 6000 provided by Udacity

For details about how I created the training data, see the next section.

###Data Augmentation and Training Strategy

#### Pre-processing:
I used a lambda layer for image normalization so that it can be performed in parallel in the GPU instead of CPU. I'm cropping model in all sides to remove unwanted details in all the sides.

Here is a example of cropped and resized image.

![alt text][image2]

#### Udacity data
Below is the angle distribution of the Udacity provided data

![alt text][image3]

As you can see the data is heavily biased towards steering angle zero. To get a better distribution I randomly used the left and right camera images by adding a 0.2 bias to the center image steering angle.

Angle distribution with left and right camera images

![alt text][image4]

#### Flipping the images
I randomly flipped the images to remove the left steering bias .

![alt text][image5]

#### Image Horizontal Translation
I trained with this data but the car failed to steer the first turn itself. In order to get a better distribution I used the image translation technique to generate additional data points based on this [article](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.11yb60izi)

The improved angle distribution after the translating image in x direction and off-setting steering angle with 0.004 units per pixel shift depending on direction.

New improved angle distribution  

![alt text][image6]

The car started performing reasonably well after training with the above augmented data.

#### Transfer learning
From here I started using started learning and started to tune the model with more data I collected. I collected data focussed on steering in curves, reverse lap and curb to center driving etc depending on the model behavior in the track. I also added a bit of data from track2 to generalize it better.

Angle distribution of one set of data collected by me.  

![alt text][image7]  


After data augmentation:  


![alt text][image8]

One important thing was keeping the number epoch to low value of around 5. An higher epoch always resulted in car going out of the track.


#### Improvements:
The car doesn't go out of the track in the autonomous mode. I even tried running the car backward and model performs well and keeps the car in the track. But as you can see in the video too often car touches the curb and recovers. I need to train it with better data so that it stays in the center mostly.

The model is not performing well in the second track. Future work involves adding shadow augmentation and image brightening techniques so that it will work in track2 also.
