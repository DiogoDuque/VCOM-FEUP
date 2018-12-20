# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 08:16:26 2018

@author: Joana Rocha
"""
import cv2
from glob import glob

#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, sys
#import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

# input image dimensions
img_rows, img_cols = 200, 200

# number of channels
img_channels = 3

#%% READ IMAGES AND RESIZE THEM

#path1 = 'C:/Users/Joana Rocha/Documents/GitHub/VCOM-FEUP/camara'    #path of folder of images 
path1 = 'camara'   
path2 = 'musica' 
path3 = 'serralves'
path4 = 'clerigos'
path5 = 'arrabida'  


images1 = glob(os.path.join(path1, '*.jpg'))
images2 = glob(os.path.join(path2, '*.jpg'))
images3 = glob(os.path.join(path3, '*.jpg'))
images4 = glob(os.path.join(path4, '*.jpg'))
images5 = glob(os.path.join(path5, '*.jpg'))

def proc_images(images,img_rows,img_cols):
    #Returns  array x of resized images: 

    x = []

    for img in images:
        base = os.path.basename(img)
    # Read and resize image
        full_size_image = cv2.imread(img)
    #x.append(full_size_image)
        x.append(cv2.resize(full_size_image, (img_cols,img_rows), interpolation=cv2.INTER_CUBIC))

    return x

resized1 = proc_images(images1,img_rows,img_cols)
resized2 = proc_images(images2,img_rows,img_cols)
resized3 = proc_images(images3,img_rows,img_cols)
resized4 = proc_images(images4,img_rows,img_cols)
resized5 = proc_images(images5,img_rows,img_cols)

#%% BUILD MATRIX WITH IMAGES AND GET LABELS
num_samples=len(resized1) + len(resized2) + len(resized3) + len(resized4) + len(resized5)

# Create list to store all images
train_images = resized1 + resized2 + resized3 + resized4 + resized5

# Create list to match images
train_labels = [None] * num_samples
for i in range(0,len(resized1)):
    train_labels[i] = path1
for i in range(len(resized1),len(resized1)+len(resized2)):
    train_labels[i] = path2
for i in range(len(resized1)+len(resized2),len(resized1)+len(resized2)+len(resized3)):
    train_labels[i] = path3
for i in range(len(resized1)+len(resized2)+len(resized3),len(resized1)+len(resized2)+len(resized3)+len(resized4)):
    train_labels[i] = path4
for i in range(len(resized1)+len(resized2)+len(resized3)+len(resized4),len(resized1)+len(resized2)+len(resized3)+len(resized4)+len(resized5)):
    train_labels[i] = path5

train_data = [train_images,train_labels]
#
#img=immatrix[167].reshape(img_rows,img_cols)
#plt.imshow(img)
#plt.imshow(img,cmap='gray')
#print (train_data[0].shape)
#print (train_data[1].shape)
#
#
##%%
#(X, y) = (train_data[0],train_data[1])
#
#
## STEP 1: split X and y into training and testing sets
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
#
#
#X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#
#X_train /= 255
#X_test /= 255
#
#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')
#
## convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)
#
#i = 100
#plt.imshow(X_train[i, 0], interpolation='nearest')
#print("label : ", Y_train[i,:])