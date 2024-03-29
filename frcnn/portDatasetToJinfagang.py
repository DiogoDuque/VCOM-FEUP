import os
import random
import numpy as np
from glob import glob
import sys
sys.path.append("..")

from annotations_parser import getXmlFilesAnnotations, convertXmlAnnotationsToArray, _displayAnnotationsString

datasetFolders = ['camara', 'musica', 'serralves', 'clerigos', 'arrabida']
datasetRootPath = os.path.join("..", "images")

imagesPaths = []
for folder in datasetFolders:
    path = os.path.join(datasetRootPath, folder)
    imagesPaths += glob(os.path.join(path, '*.jpg'))
    imagesPaths += glob(os.path.join(path, '*.jpeg'))
    imagesPaths += glob(os.path.join(path, '*.png'))
imagesLabels = np.array([x.split("/")[-1].split("\\")[-1].split("-")[0] for x in imagesPaths])
imagesPaths = np.array(imagesPaths)
imagesPaths = [os.path.join("..", path) for path in imagesPaths] # final adjustement because final dataset file is one folder level below

# get annotations
annotations = getXmlFilesAnnotations()
bboxInfos, notFoundImgs = convertXmlAnnotationsToArray(annotations, imagesPaths, False, True)

# delete images without annotations
#for notFound in notFoundImgs:
#    index = np.flatnonzero(np.core.defchararray.find(imagesPaths, notFound)!=-1)[0]
#    imagesPaths = np.delete(imagesPaths, index)
#    imagesLabels = np.delete(imagesLabels, index)

# reshape to prepare for concatenation
imagesPaths = np.reshape(imagesPaths, [len(imagesPaths), 1])
imagesLabels = np.reshape(imagesLabels, [len(imagesLabels), 1])

# concatenation all info
datasetLines = np.concatenate((imagesPaths, bboxInfos), axis=1)
datasetLines = np.concatenate((datasetLines, imagesLabels), axis=1)

# prepare as strings
datasetLinesStr = []
for line in datasetLines:
    datasetLinesStr.append(','.join(line))

# separate into train and test
print(len(datasetLinesStr))
random.seed()
lenTestElems = int(0.1*len(datasetLinesStr))
testDataset = []
for i in range(lenTestElems):
    index = random.randint(0, len(datasetLinesStr))
    elem = datasetLinesStr[index]
    testDataset.append(elem)
    del datasetLinesStr[index]
trainDataset = datasetLinesStr

# write to files
with open(os.path.join('keras_frcnn-jinfagang','kitti_simple_label.txt'), 'w') as f:
    for line in trainDataset:
        f.write("%s\n" % line)
with open(os.path.join('keras_frcnn-jinfagang','kitti_simple_label_test.txt'), 'w') as f:
    for line in testDataset:
        f.write("%s\n" % line)
