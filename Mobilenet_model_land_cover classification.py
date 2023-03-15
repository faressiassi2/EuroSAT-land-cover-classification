#!/usr/bin/env python
# coding: utf-8

# In[13]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
import matplotlib.image as mpimg
import numpy as np


# In[6]:

train_AnnualCrop_dir = os.path.join('Desktop/train_data/AnnualCrop')

train_Forest_dir = os.path.join('Desktop/train_data/Forest')

train_HerbaceousVegetation_dir = os.path.join('Desktop/train_data/HerbaceousVegetation')

train_Highway_dir = os.path.join('Desktop/train_data/Highway')

train_Industrial_dir = os.path.join('Desktop/train_data/Industrial')

train_Pasture_dir = os.path.join('Desktop/train_data/Pasture')

train_PermanentCrop_dir = os.path.join('Desktop/train_data/PermanentCrop')

train_Residential_dir = os.path.join('Desktop/train_data/Residential')

train_River_dir = os.path.join('Desktop/train_data/River')

train_SeaLake_dir = os.path.join('Desktop/train_data/SeaLake')


# In[8]:

valid_AnnualCrop_dir = os.path.join('Desktop/valid_data/AnnualCrop')

valid_Forest_dir = os.path.join('Desktop/valid_data/Forest')

valid_HerbaceousVegetation_dir = os.path.join('Desktop/valid_data/HerbaceousVegetation')

valid_Highway_dir = os.path.join('Desktop/valid_data/Highway')

valid_Industrial_dir = os.path.join('Desktop/valid_data/Industrial')

valid_Pasture_dir = os.path.join('Desktop/valid_data/Pasture')

valid_PermanentCrop_dir = os.path.join('Desktop/valid_data/PermanentCrop')

valid_Residential_dir = os.path.join('Desktop/valid_data/Residential')

valid_River_dir = os.path.join('Desktop/valid_data/River')

valid_SeaLake_dir = os.path.join('Desktop/valid_data/SeaLake')


# In[9]:
test_AnnualCrop_dir = os.path.join('Desktop/test_data/AnnualCrop')

test_Forest_dir = os.path.join('Desktop/test_data/Forest')

test_HerbaceousVegetation_dir = os.path.join('Desktop/test_data/HerbaceousVegetation')

test_Highway_dir = os.path.join('Desktop/test_data/Highway')

test_Industrial_dir = os.path.join('Desktop/test_data/Industrial')

test_Pasture_dir = os.path.join('Desktop/test_data/Pasture')

test_PermanentCrop_dir = os.path.join('Desktop/test_data/PermanentCrop')

test_Residential_dir = os.path.join('Desktop/test_data/Residential')

test_River_dir = os.path.join('Desktop/test_data/River')

test_SeaLake_dir = os.path.join('Desktop/test_data/SeaLake')


# In[7]:
print('total training AnnualCrop images:', len(os.listdir(train_AnnualCrop_dir)))
print('total training Forest images:', len(os.listdir(train_Forest_dir)))
print('total training HerbaceousVegetation images:', len(os.listdir(train_HerbaceousVegetation_dir)))
print('total training Highway images:', len(os.listdir(train_Highway_dir)))
print('total training Industrial images:', len(os.listdir(train_Industrial_dir)))
print('total training Pasture images:', len(os.listdir(train_Pasture_dir)))
print('total training PermanentCrop images:', len(os.listdir(train_PermanentCrop_dir)))
print('total training Residential images:', len(os.listdir(train_Residential_dir)))
print('total training River images:', len(os.listdir(train_River_dir)))
print('total training SeaLake images:', len(os.listdir(train_SeaLake_dir)))


# In[10]:
print('total validation AnnualCrop images:', len(os.listdir(valid_AnnualCrop_dir)))
print('total validation Forest images:', len(os.listdir(valid_Forest_dir)))
print('total validation HerbaceousVegetation images:', len(os.listdir(valid_HerbaceousVegetation_dir)))
print('total validation Highway images:', len(os.listdir(valid_Highway_dir)))
print('total validation Industrial images:', len(os.listdir(valid_Industrial_dir)))
print('total validation Pasture images:', len(os.listdir(valid_Pasture_dir)))
print('total validation PermanentCrop images:', len(os.listdir(valid_PermanentCrop_dir)))
print('total validation Residential images:', len(os.listdir(valid_Residential_dir)))
print('total validation River images:', len(os.listdir(valid_River_dir)))
print('total validation SeaLake images:', len(os.listdir(valid_SeaLake_dir)))


# In[11]:
print('total testing AnnualCrop images:', len(os.listdir(test_AnnualCrop_dir)))
print('total testing Forest images:', len(os.listdir(test_Forest_dir)))
print('total testing HerbaceousVegetation images:', len(os.listdir(test_HerbaceousVegetation_dir)))
print('total testing Highway images:', len(os.listdir(test_Highway_dir)))
print('total testing Industrial images:', len(os.listdir(test_Industrial_dir)))
print('total testing Pasture images:', len(os.listdir(test_Pasture_dir)))
print('total testing PermanentCrop images:', len(os.listdir(test_PermanentCrop_dir)))
print('total testing Residential images:', len(os.listdir(test_Residential_dir)))
print('total testing River images:', len(os.listdir(test_River_dir)))
print('total testing SeaLake images:', len(os.listdir(test_SeaLake_dir)))


# In[12]:
img = mpimg.imread('Desktop/train_data/AnnualCrop/AnnualCrop_1.jpg')
print(img.shape)
imgplot = plt.imshow(img)

# In[14]:
#Using the MobileNet model architecture by using pre-trained weights on ImageNet dataset instead of using random initialization
# of the weights and we will fine tune this model by not including the fully connected layers and using our fully connected 
# layers adapted to our project by setting include_top=False.

model     = MobileNet(weights='imagenet', include_top=False, input_shape=(64,64,3)) 
x         = model.output
x         = GlobalAveragePooling2D()(x) #Modification possible : GlobalMaxPooling2D()
x         = Dense(128,activation='relu')(x)
x         = Dense(512,activation='relu')(x)
x         = Dense(1024,activation='relu')(x)
sortie    = Dense(10, activation='softmax')(x)
mobilenet = Model(inputs=model.input, outputs=sortie)


# In[ ]:
mobilenet.summary()


# In[15]:
# We will use the data augmentation to avoid the overfitting problem:

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1/255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
# flow_from_directory=Takes the path to a directory & generates batches of augmented data.
train_generator = train_datagen.flow_from_directory(
        'Desktop/train_data',  # This is the source directory for training images
        target_size=(64, 64),  # All images will be resized to 64x64
        batch_size=128, # Size of the batches of data (default: 32).
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        'Desktop/valid_data',  
        target_size=(64, 64),  
        batch_size=32,
        class_mode='categorical')


# In[16]:
#compile the model:
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adamax

mobilenet.compile(
    loss = 'categorical_crossentropy',
    optimizer = Adam(lr=0.001) ,
    metrics = ['accuracy']
)


# In[18]:
history = mobilenet.fit(
    train_generator,
    epochs = 10,
    validation_data = validation_generator
)


# In[20]:
# Plot the loss and accuracy curves for training and validation to know if we overfitting or not:
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(acc,label='Training accuracy')
plt.plot(val_acc,label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(loss,label='Training Loss')
plt.plot(val_loss,label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[22]:


#evaluate the model on the validation data:
results = mobilenet.evaluate(validation_generator, verbose=0)
print("test loss, test acc:", results)


# In[23]:
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, classification_report


# In[24]:
y_pred=mobilenet.predict(validation_generator)


# In[25]:
y_pred_modi = np.argmax(y_pred, axis=1)


# In[26]:
print(confusion_matrix(validation_generator.classes, y_pred_modi))


# In[27]:
print(classification_report(validation_generator.classes, y_pred_modi))


# In[28]:
#saving the model:
from keras.models import load_model
mobilenet.save('model_mobilenet_land cover_file.h5')


# In[29]:
my_model_mobilenet = load_model('model_mobilenet_land cover_file.h5')


# In[ ]:
# predicting on the testing data:
# In[31]:
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
        'Desktop/test_data',  
        target_size=(64, 64),  
        class_mode='categorical')


# In[32]:
predictions = mobilenet.predict(test_generator)

# In[33]:
predictions_modi = np.argmax(predictions, axis=1)


# In[34]:
for i, j in enumerate(predictions_modi[:20]):
    print("Actual_classe: {}".format(test_generator.classes[i]), "Predicted_classe: {}".format(j))







