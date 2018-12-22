import os
import numpy as np
from glob import glob
import sys
sys.path.append("..")

from annotations_parser import getXmlFilesAnnotations, convertXmlAnnotationsToArray, _displayAnnotationsString

datasetFolders = ['camara', 'musica', 'serralves', 'clerigos', 'arrabida']
datasetRootPath = ".."

imagesPaths = []
for folder in datasetFolders:
    path = os.path.join(datasetRootPath, folder)
    imagesPaths += glob(os.path.join(path, '*.jpg'))
    imagesPaths += glob(os.path.join(path, '*.jpeg'))
    imagesPaths += glob(os.path.join(path, '*.png'))
imagesLabels = np.array([x.split("/")[-1].split("\\")[-1].split("-")[0] for x in imagesPaths])
imagesPaths = np.array(imagesPaths)

annotations = getXmlFilesAnnotations()
bboxInfos = np.array(convertXmlAnnotationsToArray(annotations, imagesPaths))
print(bboxInfos)
print("======================")
bboxInfos = [",".join(item) for item in bboxInfos.astype(str)]
print(bboxInfos)

# trying to concat into format: /path/training/image_2/000001.png,599.41,156.40,629.75,189.25,Truck


