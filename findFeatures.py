# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 08:33:26 2019

@author: Joana Rocha
"""

import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from joblib import dump,load
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

from glob import glob

# input image dimensions
img_rows, img_cols = 200, 200

# number of channels
img_channels = 3

# Get the path of the training set
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

training_names = images1 + images2 + images3 + images4 + images5

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
num_samples = len(resized1) + len(resized2) + len(resized3) + len(resized4) + len(resized5)

# Get the training classes names and store them in a list
train_images = resized1 + resized2 + resized3 + resized4 + resized5

# Create list to match images
train_labels = [None] * num_samples
for i in range(0,len(resized1)):
    train_labels[i] = 1 #path1
for i in range(len(resized1),len(resized1)+len(resized2)):
    train_labels[i] = 2 #path2
for i in range(len(resized1)+len(resized2),len(resized1)+len(resized2)+len(resized3)):
    train_labels[i] = 3
for i in range(len(resized1)+len(resized2)+len(resized3),len(resized1)+len(resized2)+len(resized3)+len(resized4)):
    train_labels[i] = 4
for i in range(len(resized1)+len(resized2)+len(resized3)+len(resized4),len(resized1)+len(resized2)+len(resized3)+len(resized4)+len(resized5)):
    train_labels[i] = 5

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
#image_paths = []
#image_classes = []
#class_id = 0
#for training_name in training_names:
#    dir = os.path.join(train_path, training_name)
#    class_path = imutils.imlist(dir)
#    image_paths+=class_path
#    image_classes+=[class_id]*len(class_path)
#    class_id+=1
    
image_paths = training_names
image_classes = train_labels
#class_id = 4   

# Create feature extraction and keypoint detector objects
#fea_det = cv2.FeatureDetector_create("SIFT")
#des_ext = cv2.DescriptorExtractor_create("SIFT")
fea_det = cv2.KAZE_create()


# List where all the descriptors are stored
des_list = []

for im in original_images:
    kpts, des = fea_det.detectAndCompute(im, None)
    des_list.append((im, des))   
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

# Perform k-means clustering
k = 100
voc, variance = kmeans(descriptors, k, 1) 

# Calculate the histogram of features
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Train the Linear SVM
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))

# Save the SVM
#joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)  
#dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)     

dump((clf, train_labels, stdSlr, k, voc), "bof.pkl", compress=3)   

#TENHO DE FAZER UMA PASTA A SEPARAR TRAIN E TEST!