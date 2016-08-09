import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import pandas as pd
import numpy as np

df_train = pd.read_csv('./data/train.csv', header=0)
df_test = pd.read_csv('./data/test.csv', header=0)

labels = pd.DataFrame(pd.get_dummies(df_train['label']))
df_train = df_train.drop(['label'], axis=1)


msk = np.random.rand(len(df_train)) < 0.8

train_data = df_train[msk].as_matrix().reshape([-1, 28, 28, 1])
train_labels = labels[msk].as_matrix()

validation_data = df_train[~msk].as_matrix().reshape([-1, 28, 28, 1])
validation_labels = labels[~msk].as_matrix()

test = df_test.as_matrix().reshape([-1, 28, 28, 1])

del df_train
del df_test
del msk

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=1e-5, loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': train_data}, {'target': train_labels}, n_epoch=10,
          validation_set=({'input': validation_data}, {'target': validation_labels}), show_metric=True, run_id='convnet_mnist')

predictions = None
for chunk in np.split(test, 4):
    if predictions is None:
        predictions = model.predict(chunk)
    else:
        predictions = np.append(predictions, model.predict(chunk), axis=0)

predictions = pd.DataFrame(predictions).idxmax(axis=1)
predictions.index += 1
predictions.to_csv('./data/predictions.csv', header=['Label'], index=True, index_label='ImageId')
