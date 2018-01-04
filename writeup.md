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

[image1]: ./bars_train.png "Visualization"
[image2]: ./data/streetsign_vorfahrt.png "Traffic Sign 1"
[image3]: ./data/streetsign_nopassing.png "Traffic Sign 2"
[image4]: ./data/streetsign_yield.png "Traffic Sign 3"
[image5]: ./data/streetsign_forward.png "Traffic Sign 4"
[image6]: ./data/streetsign_tempolimit60.png "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it! Here is a link to my [project](https://github.com/xlani/CarND-Traffic-Sign-Classifier-Project/).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images.
* The size of the validation set is 4410 images.
* The size of test set is 12630.
* The shape of a traffic sign image is 32x32.
* The number of unique classes/labels in the data set is 43.

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed along the different classes. You see the distribution of the training data.

![alt text][image1]

If you want to see more, have a look into the jupyter notebook file.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?

I tried normalizing and grayscaling. Grayscaling actually seemed to make results worse. So I just used normalization (function cv2.normalize) for preprocessing the images.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer | Description	|
|:---------------------:|:----------------------------------------------------------:|
| Input | 32x32x3 RGB image (normalized) |
| Convolution 5x5 | 1x1 stride, valid padding, depth 6, output 28x28x6 |
| RELU | |
| Max pooling	| 2x2 stride, output 14x14x6 |
| Dropout	| |
| Convolution 5x5 | 1x1 stride, valid padding, depth 16, output 10x10x16 |
| RELU | |
| Max pooling	| 2x2 stride,  output 5x5x16 |
| Dropout	| |
| Flatten	| from (5,5,16) to 400 |
| Fully connected	| input = 400, output = 120 |
| RELU | |
| Dropout	| |
| Fully connected	| input = 120, output = 84 |
| RELU | |
| Dropout	| |
| Fully connected | input = 84, output = 43	|
| Softmax	| |

#### 3. Describe how you trained your model.

See section 4 for a combined answer for sections 3 & 4.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started with LeNet architecture as recommended in the project introduction (adjusted for input of RGB images and number of labels) which got me 99% training accuracy and around 90% validation accuracy after 10 epochs of training. The function 'evaluate' for calculating the accuracies can be seen in cell 8.

Because high accuracy on the training set but low accuracy on the validation set indicates over fitting, I decided to include dropout layers in the LeNet architecture. From that point on I tried around with values for the hyperparameter KEEP_RATE (rate of not dropping weights for dropout regularization). I started with KEEP_RATE = 0.5 and went up to 0.95 (for LEARN_RATE = 0.001 & EPOCHS = 50). The results have shown that the validation accuracy went up with the KEEP_RATE. So I decided to stick with KEEP_RATE = 0.95.

After that I gave a try to increase the learning rate to LEARN_RATE = 0.002. That lead to much faster increase of the validation accuracy. Therefore I decided to implement an early stopping approach. I decided to stop learning if reaching validation accuracy >= 94%, combined with training accuracy >= 99%. In my final training run, the early stopping criteria was actually reached in the final epoch 40 unlike earlier runs that stopped at epochs 20-25.

My final model results were (see cell 9 & 10):
* training set accuracy of 99.6%
* validation set accuracy of 94.2%
* test set accuracy of 92.5%

With the following hyperparameters (see cell 5):

* LEARN_RATE = 0.002
* EPOCHS = 40 (max value, because of implemented stopping criteria)
* BATCH_SIZE = 128
* KEEP_RATE = 0.95

Implemented but non used hyperparameters
* BETA = 0. for L2 regularization
* RELU_BIAS = 0. - bias to possibly activate all ReLu nodes at the first training step

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are the six German traffic signs that I found on the web and used for testing my model:

![alt text][image2] ![alt text][image3] ![alt text][image4]
![alt text][image5] ![alt text][image6]

All of them are of good quality (good brightness, no hidden parts) and expected to be classified correctly.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image | Prediction |
|:---------------------:|:---------------------------------------------:|
| Priority road | Priority road |
| No passing | No passing |
| Yield	| Yield	|
| Only forward | Only forward	|
| Speed limit (60km/h) | Speed limit (60km/h) |

The model was able to correctly classify five of the five traffic sign images, which gives an accuracy of 100%. The accuracy of the test set was 92.2%. Because of the low number of the additionally tested image, it is hard to compare these numbers. In addition all of the five tested images contained signs that are quite well represented in the training data.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The model was very sure for all of the 5 images with over 99% confidence.

For the first image, the model was very sure that this is a stop sign and the image does contain a stop sign. The top five soft max probabilities were

| Probability | Prediction |
|:---------------------:|:---------------------------------------------:|
| .994 | Priority road |
| .003 | End of no passing |
| .003 | Keep right |
| .000	|	Beware of ice/snow |
| .000	| Roundabout mandatory |

For the second image, the model was very sure that this is a no passing sign and the image does contain a no passing sign. The top five soft max probabilities were

| Probability | Prediction |
|:---------------------:|:---------------------------------------------:|
| 1. | No passing |
| .0 | No passing for vehicles over 3.5 metric tons |
| .0 | Vehicles over 3.5 metric tons prohibited |
| .0 | Speed limit (60km/h) |
| .0 | End of no passing |

For the third image, the model was very sure that this is a yield sign and the image does contain a yield sign. The top five soft max probabilities were

| Probability | Prediction |
|:---------------------:|:---------------------------------------------:|
| 1. | Yield |
| .0 | No passing |
| .0 | Speed limit (60km/h) |
| .0 | Speed limit (80km/h) |
| .0 | Children crossing |

For the fourth image, the model was very sure that this is an ahead only sign and the image does contain an ahead only sign. The top five soft max probabilities were

| Probability | Prediction |
|:---------------------:|:---------------------------------------------:|
| 1. | Ahead only |
| .0 | Turn right ahead |
| .0 | Go straight or right |
| .0 | Yield |
| .0 | No passing for vehicles over 3.5 metric tons |

For the fifth image, the model was very sure that this is a speed limit (60km/h) and the image does contain a speed limit (60km/h) sign. The top five soft max probabilities were

| Probability | Prediction |
|:---------------------:|:---------------------------------------------:|
| 0.994 | Speed limit (60km/h) |
| .006 | Speed limit (30km/h) |
| .000 | Speed limit (50km/h) |
| .000 | Speed limit (80km/h) |
| .000 | Roundabout mandatory |
