# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 08:16:26 2018

@author: Joana Rocha
"""
import cv2
from glob import glob
from xml.dom import minidom

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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from annotations_parser import getXmlFilesAnnotations, convertXmlAnnotationsToArray

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
images1 = images1 + glob(os.path.join(path1, '*.jpeg'))
images1 = images1 + glob(os.path.join(path1, '*.png'))
images2 = glob(os.path.join(path2, '*.jpg'))
images2 = images2 + glob(os.path.join(path2, '*.jpeg'))
images2 = images2 + glob(os.path.join(path2, '*.png'))
images3 = glob(os.path.join(path3, '*.jpg'))
images3 = images3 + glob(os.path.join(path3, '*.jpeg'))
images3 = images3 + glob(os.path.join(path3, '*.png'))
images4 = glob(os.path.join(path4, '*.jpg'))
images4 = images4 + glob(os.path.join(path5, '*.jpeg'))
images4 = images4 + glob(os.path.join(path5, '*.png'))
images5 = glob(os.path.join(path5, '*.jpg'))
images5 = images5 + glob(os.path.join(path5, '*.jpeg'))
images5 = images5 + glob(os.path.join(path5, '*.png'))

def proc_images(images,img_rows,img_cols):
    #Returns  array x of resized images: 

    x = []
    original_images = []

    for img in images:
        base = os.path.basename(img)
    # Read and resize image
        full_size_image = cv2.imread(img)
        original_images.append(full_size_image)
    #x.append(full_size_image)
        x.append(cv2.resize(full_size_image, (img_cols,img_rows), interpolation=cv2.INTER_CUBIC))

    return x, original_images

resized1, original_images1 = proc_images(images1,img_rows,img_cols)
resized2, original_images2 = proc_images(images2,img_rows,img_cols)
resized3, original_images3 = proc_images(images3,img_rows,img_cols)
resized4, original_images4 = proc_images(images4,img_rows,img_cols)
resized5, original_images5 = proc_images(images5,img_rows,img_cols)

original_images = original_images1 + original_images2 + original_images3 + original_images4 + original_images5
#%% BUILD MATRIX WITH IMAGES AND GET LABELS
num_samples = len(resized1) + len(resized2) + len(resized3) + len(resized4) + len(resized5)

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

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)



#%% GET BOUDING BOX COORDINATES


# LER ANOTAÇOES E TIRAR X, Y, W e H DE CADA BBOX
# ASSOCIAR CADA ANOTAÇAO A IMAGEM CORRESPONDENTE PELA MESMA ORDEM Q TRAIN IMAGES E TRAIN LABELS

# substituir 'filenames' por uma lista dos ficheiros. a funcao tanto aceita nomes de ficheiros do tipo "camara/camara-0000.jpg" como só "camara-0000.jpg" :)
filenames = ["camara/camara-0000.jpg", "camara/camara-0001.jpg", "camara-0002.jpg"]
annotations = getXmlFilesAnnotations()
bboxes = convertXmlAnnotationsToArray(annotations, filenames, True, False, True)
print(bboxes)

#%% RESIZE BBOXES (TO MATCH RESIZED IMAGES)
resized_bboxes = bboxes

for i in range(len(original_images)):
    resized_bboxes[i][0] = (bboxes[i][0] * img_rows) / (original_images[i][0]) #z
    resized_bboxes[i][1] = (bboxes[i][1] * img_cols) / (original_images[i][1]) #w
    resized_bboxes[i][2] = (bboxes[i][2] * img_cols) / (original_images[i][1]) #x
    resized_bboxes[i][3] = (bboxes[i][3] * img_rows) / (original_images[i][0]) #y


#%% ONE HOT VECTORS

label_encoder = LabelEncoder()
valores_numericos = label_encoder.fit_transform(train_labels)  #string to num
#label_encoder.inverse_transform() se quiser passar p str outra vez
print(valores_numericos)

onehot_encoder = OneHotEncoder(sparse=False)
inteiros = valores_numericos.reshape(len(valores_numericos),1)
train_labels = onehot_encoder.fit_transform(inteiros)

print(train_labels)

#%% DATA AUGMENTATION


#%% CONCATENATE LABELS AND BBOXES
train_labels_bbox = np.concatenate([bboxes, train_labels], axis=-1).reshape(num_samples, -1)
print(train_labels_bbox.shape)

#%% MODEL

i = int(0.8 * num_samples)
training_images = train_images[:i] #divide as imagens em train e test
testing_images = train_images[i:]
training_y = train_labels_bbox[:i]
testing_y = train_labels_bbox[i:]
test_bboxes = bboxes[i:]

model = Sequential()
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(train_labels_bbox.shape[-1]))