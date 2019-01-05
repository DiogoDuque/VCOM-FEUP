import os
import pydot_ng as pydot
from keras_frcnn import config
import pickle
from keras.models import Model
from keras.layers import Input
from keras.utils import plot_model
import keras_frcnn.resnet as nn

with open('config.pickle', 'rb') as f_in:
    cfg = pickle.load(f_in)
cfg.use_horizontal_flips = False
cfg.use_vertical_flips = False
cfg.rot_90 = False

class_mapping = cfg.class_mapping
if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
input_shape_img = (None, None, 3)
input_shape_features = (None, None, 1024)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(cfg.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)
classifier = nn.classifier(feature_map_input, roi_input, cfg.num_rois, nb_classes=len(class_mapping),  trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

model_rpn.load_weights(cfg.model_path, by_name=True)
model_classifier.load_weights(cfg.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

# export to png

plot_model(model_rpn, to_file='rpn.png')
plot_model(model_classifier, to_file='classifier.png')
plot_model(model_rpn, show_shapes=True, to_file='rpn-shapes.png')
plot_model(model_classifier, show_shapes=True, to_file='classifier-shapes.png')

#history_rpn = model_rpn.fit()
history_class = model_classifier.fit()
print(history_class)