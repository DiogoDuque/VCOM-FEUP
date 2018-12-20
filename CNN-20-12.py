# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:14:38 2018

@author: Catarina Dias
"""

import cv2
from glob import glob
import os

#KERAS
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import numpy as np 
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils import to_categorical


# input image dimensions
img_rows, img_cols = 200, 200

# number of channels
img_channels = 3

# READ IMAGES AND RESIZE THEM

#path1 = 'C:/Users/Joana Rocha/Documents/GitHub/VCOM-FEUP/camara'    #path of folder of images 
path1 = 'camara'   
path2 = 'musica' 
path3 = 'serralves'
path4 = 'clerigos'
path5 = 'arrabida'  

#images1=glob(path1+'/'+ '*.')
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

    for img in images:
        #image name
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

# BUILD MATRIX WITH IMAGES AND GET LABELS
num_samples=len(resized1) + len(resized2) + len(resized3) + len(resized4) + len(resized5)

# Create list to store all images
train_images = resized1 + resized2 + resized3 + resized4 + resized5

# Create list to match images
train_labels = [None] * num_samples
for i in range(0,len(resized1)):
    train_labels[i] = 0
for i in range(len(resized1),len(resized1)+len(resized2)):
    train_labels[i] = 1
for i in range(len(resized1)+len(resized2),len(resized1)+len(resized2)+len(resized3)):
    train_labels[i] = 2
for i in range(len(resized1)+len(resized2)+len(resized3),len(resized1)+len(resized2)+len(resized3)+len(resized4)):
    train_labels[i] = 3
for i in range(len(resized1)+len(resized2)+len(resized3)+len(resized4),len(resized1)+len(resized2)+len(resized3)+len(resized4)+len(resized5)):
    train_labels[i] = 4

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)
print(train_labels)


#Convert to categorical data
train_labels=to_categorical(train_labels, num_classes=5)

#splitting the data into training and testing
X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size=0.2 ) # training to testing ratio is 0.8:0.2

print("Number of train examples:", X_train.shape[0])
print("Number of validation examples:", X_val.shape[0])


#-----------------------------------------------------------------------------------------

#Load the VGG model- without the top layer ( which consists of fully connected layers )
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(train_images.shape[1], train_images.shape[2], 3))


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


#DATA AUGMENTATION

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 20
val_batchsize = 10


train_generator = train_datagen.flow(X_train, Y_train, shuffle=True, seed=10, batch_size=train_batchsize)


val_generator = validation_datagen.flow(X_val, Y_val, shuffle=True, seed=10, batch_size=train_batchsize)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=None,
      epochs=15,
      validation_data=val_generator,
      shuffle=True,
      verbose=2)
 
# Save the model
model.save('small_last4.h5')


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()
