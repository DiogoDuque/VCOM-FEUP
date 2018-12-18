# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:14:38 2018

@author: Catarina Dias
"""

from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import os 
import numpy as np 
from keras.preprocessing import image
from sklearn.cross_validation import train_test_split
 
#Import data
PATH = os.getcwd()


train_path = PATH+'/data/'
train_batch = os.listdir(train_path)
x_train = []
 
# if data are in form of images-- ISTO DEVE SER AJUSTADO CONSOANTE O NOME DAS NOSSAS IMAGENS
for sample in train_data:
	img_path = train_path+sample
	x = image.load_img(img_path)
	# preprocessing if required
	x_train.append(x)
 
# Convert list into numpy array
x_train = np.array(x_train)


#FAZER IMPORT TAMBÉM DAS LABELS


#splitting the data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(x_train, Y, test_size=0.2 ) # training to testing ratio is 0.8:0.2


#-----------------------------------------------------------------------------------------

#Load the VGG model- without the top layer ( which consists of fully connected layers )
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))


#Freeze the layers except the last 4 layers
#Each layer has a parameter called “trainable”. 
#For freezing the weights of a particular layer, we should set this parameter to False, indicating that this layer should not be trained.

for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
 
# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)
    
    
# Create a new model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)
 
# Add new layers - add a fully connected layer followed by a softmax layer with 5 outputs 
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()


