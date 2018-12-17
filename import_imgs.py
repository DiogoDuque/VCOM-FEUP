# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 09:47:41 2018

@author: Joana Rocha
"""

from glob import glob
import numpy
import cv2
from skimage.transform import resize
 
#mypath='C:/Users/Joana Rocha/Documents/GitHub/VCOM-FEUP/imagens'
x_train = numpy.zeros((400,600,75))
y_train = numpy.zeros((1,25))
y_train[0,0:5]=1
y_train[0,5:11]=2
y_train[0,11:16]=3
y_train[0,16:21]=4
y_train[0,21:26]=5

i=0
j=3
for fn in glob('*.jpg'):
    img = cv2.imread(fn)
    
    cv2.imshow('imported image (RGB)',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    image_resized = cv2.resize(img, (600, 400))
    
    cv2.imshow('resized image (RGB)',image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    x_train[:,:,i:j] = image_resized
    i=i+3
    j=j+3
    


    
## the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#
#x_train = x_train.reshape(60000,28,28,1)
#x_test = x_test.reshape(10000,28,28,1)