
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.normalization import local_response_normalization

import pandas as pd
import numpy as np
import os
import cv2

def load_bmps_from_folder(folder):
    bmps = []
    for root, dirs, filenames in os.walk(folder):
        for f in filenames:
            bmps.append(cv2.imread(os.path.join(root, f), 0))

    return np.array(bmps, dtype=np.float64)

xs_train = load_bmps_from_folder('./data/trainResized')
ys_train = pd.get_dummies(pd.read_csv('./data/trainLabels.csv', header=0)['Class'])
ys_map = pd.Series(ys_train.columns.values)
xs_test = load_bmps_from_folder('./data/testResized')

msk = np.random.rand(xs_train.shape[0]) < 0.8

train_data = xs_train[msk].reshape([-1, 20, 20, 1])
train_labels = ys_train[msk].as_matrix()

validation_data = xs_train[~msk].reshape([-1, 20, 20, 1])
validation_labels = ys_train[~msk].as_matrix()

del xs_train
del ys_train

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=10.)
img_aug.add_random_blur()

# Building convolutional network
network = input_data(shape=[None, 20, 20, 1], name='input',
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = fully_connected(network, 2048, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 2048, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 62, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': train_data}, {'target': train_labels}, n_epoch=100,
          validation_set=({'input': validation_data}, {'target': validation_labels}),
          show_metric=True, run_id='convnet_mnist')

pred_matrix = model.predict(xs_test.reshape([-1, 20, 20, 1]))
idx = pd.DataFrame(pred_matrix).idxmax(axis=1)
pred_labels = ys_map[idx]
pred_labels.to_csv('./data/predictions.csv', header=['Label'], index=True, index_label='ImageId')