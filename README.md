# EuroSAT-land-cover-classification

We will train a Machine Learning model on the EuroSAT land cover classification dataset.
Specifically we will use the Convolutional Neural network or CNN to this multi-classification task because here we have a dataset of images and the CNNs are very powerful to extracting the features of images because it use the filtres and convolution to do that.
pre-trained weights on ImageNet dataset instead of using random initialization
# of the weights and we will fine tune this model by not including the fully connected layers and using our fully connected 
# layers adapted to our project by setting include_top=False.
This is the steps that I have used in this project:
1- We will start by gathering the dataset of images.

2-Preparing the dataset by splitting our dataset to 3 main parts : Training data for training our model, second validation data to evaluate our model on the unseen validation data and fine tuning the hyperparameters of the CNN model like learning rate, number of neurons and layers and the mini-batch size, third the testing data to test finally our  model after evaluation and hyperparameters tuning.
And also check the number of each of the classes in our dataset to verify if we have imbalanced dataset and then define the metrics to evaluate our model like accuracy, confusion_matrix, classification_report, recall, f1-score.

3-Define our model architecture, here we are using Convolutional Neural Network because we have dataset of images. We will define our custom CNN model architecture by defining a lot of layers of Convolutions and Maxpooling and Dense layers in buttom of the model. In general the architectures of a lot of CNN model is composed by a lot of convolutional layers followed by Maxpooling layers to make the features extractions of the images and finally the dense layers to make the classification task.
I will also try to use pre-defined MobileNet model architecture by using a pre-trained weights on ImageNet dataset instead of using random initialization of the weights and we will fine tune this model by not including the fully connected layers and using our fully connected layers adapted to our project by setting include_top=False.

4-We compile the model by defining an optimizer like Adam, RMSprop or SGD to update our model parameters in the training Then we define the loss used in the training and here we will use categorical_crossentropy because we have multiclass-classification problem.

5-Train our model for a certain number of epochs for example 20 or 30.

6-Ploting the loss and accuracy curves for training and validation to know if we overfitting or not.

7-Evaluate our model using the validation dataset images by using the metrics of accuracy, confusion_matrix, classification_report, recall, f1-score.

8-Hyperparameter optimization or tuning of the model by chosing a set of optimal hyperparameters of the model like learning rate , number of neurons and layers, optimizer and batch size.
In the case when we have overfitting we can use one of this methods: Dropout, Data augmentation, L2 regularization and Early stopping.

9-Finally we will generate predictions using our test dataset images.

10-Analyze the results of the 2 models used to this task:
  For the custom model architecture I did not get good results in the finall predictions used the test dataset images of 20 images.
  But for the MobileNet model architecture that I have fine tune for our task here I get a good results in the finall predictions used the test dataset images of 20    images.
