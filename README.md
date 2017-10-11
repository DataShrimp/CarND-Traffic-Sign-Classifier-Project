# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./tests/visualize.png "Visualization"
[image2]: ./tests/gray.png "Grayscaling"
[image4]: ./tests/test0.jpg "Traffic Sign 1"
[image5]: ./tests/test1.jpg "Traffic Sign 2"
[image6]: ./tests/test2.jpg "Traffic Sign 3"
[image7]: ./tests/test3.jpg "Traffic Sign 4"
[image8]: ./tests/test4.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is the readme file for the Udacity CarND Traffic Sign Recognition project and here is a link to my [project code](https://github.com/DataShrimp/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute according to the labels in train and test data set. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the value of each pixel represents the intensity information instead of color information, which can avoid the lighting effects in data sets. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it will help to quickly converge during the learning process. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6		     		|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU 					|         										|
| Max pooling 			| 2x2 stride, outputs 5x5x16					|
| Convolution 2x2 		| 1x1 stride, valid padding, outputs 4x4x32		|
| RELU 					| 												|
| Flattern				| outputs 512  									|
| Fully Connected		| outputs 256 									|
| RELU 					| 												|
| Dropout 				| 												|
| Fully Connected 		| outputs 128									|
| RELU 					|												|
| Dropout 				|												|
| Fully Connected 		| outputs 64									|
| Sigmoid				| 	        									|
| Fully Connected 		| outputs 43									|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an improved LeNet network which is more deeper than traditional one. I used three convolution layers and three fully connected layers in the network. I also add two dropout layers to avoid the overfitting problem. The optimizer is AdamOptimizer and the batch size is 128. I tried different hyperparameters and found that the learning rate is 0.001, the keep probility is 0.8 and the epochs is 15 will be fine in the project. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.956
* test set accuracy of 0.943

If a well known architecture was chosen:
* What architecture was chosen?
I chose the LeNet architecture for this project. 

* Why did you believe it would be relevant to the traffic sign application?
I think the traffic sign classifed problem is similar to the minist handwrite problem. However, it may be more difficult than the latter one for images are captured from the real environment and may be affected by illusion etc. Therefore, I added more layers to make the network deeper and added the dropout layers to avoid the overfitting problem. 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The validation set accuracy is 0.956, which is closed to the test set accuracy 0.943. It showed that the model is working well in the project. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The third image might be difficult to classify because the Road work sigh may be confused with ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Right-of-way   		| Right-of-way at the next intersection			|
| Road work				| Bicycles crossing								|
| 30 km/h	      		| 30 km/h	 					 				|
| Priority Road			| Priority Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94.3%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (weight of 9.69), and the image does contain a stop sign. The top five weights were

| Weights 	         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.69         			| 14: Stop sign   								| 
| 0.48     				| 36: Go straight or right						|
| -0.17					| 34: Turn left ahead							|
| -0.19	      			| 17: No entry					 				|
| -0.27				    | 38: Keep right      							|


For the second image, the model is relatively sure that this is a right-of-way sign (weight of 9.88), and the image does contain a right-of-way sign. The top five weights were

| Weights 	         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.88         			| 11: Right-of-way at the next intersection		| 
| 1.70     				| 30: Beware of ice/snow						|
| 1.09					| 18: General caution							|
| 0.66	      			| 27: Pedestrians				 				|
| 0.005				    | 21: Double curve     							|

For the third image, the model is confused that this is a bicycles crossing sign (weight of 4.86), and the image does contain a actual road work sign (weight of 3.95). The top five weights were

| Weights 	         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 4.86         			| 29: Bicycles crossing							| 
| 3.95     				| 25: Road work 								|
| 0.71					| 17: No entry									|
| 0.66	      			| 39: Keep left					 				|
| 0.60				    | 30: Beware of ice/snow						|

For the fourth image, the model is relatively sure that this is a 30km/h sign (weight of 9.45), and the image does contain a 30km/h sign. The top five weights were


| Weights 	         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.45         			| 1: 30km/h 	   								| 
| 1.77     				| 0: 20km/h										|
| 0.87					| 2: 50km/h										|
| 0.57	      			| 40: Roundabout mandatory		 				|
| -0.54				    | 5: 80km/h 	      							|


For the fifth image, the model is relatively sure that this is a priority road sign (weight of 10.1), and the image does contain a priority road sign. The top five weights were

| weights 	         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 10.1         			| 12: Priority road 							| 
| 0.76     				| 15: No vehicles 								|
| 0.66					| 32: End of all speed and passing limits 		|
| 0.41	      			| 40: Roundabout mandatory 		 				|
| 0.3				    | 13: Yield 	      							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I visualized the output of three convolution layers. The first layer focused on the texture features, the second layer extracted the shape and shading pattern, and the third layer could get the higher features, something like the structure infomation. 
