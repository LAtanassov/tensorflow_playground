import pandas as pd
import numpy as np
import tflearn

df_train = pd.read_csv('./data/train.csv', header=0)
df_test = pd.read_csv('./data/test.csv', header=0)

labels = pd.DataFrame(pd.get_dummies(df_train['label']))
df_train = df_train.drop(['label'], axis=1)


# Build neural network
net = tflearn.input_data(shape=[None, 784])
net = tflearn.fully_connected(net, 256)
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(df_train.as_matrix(), labels.as_matrix(), n_epoch=1, batch_size=16, show_metric=True)

predictions = pd.DataFrame(model.predict(df_test)).idxmax(axis=1)
predictions.index += 1
predictions.to_csv('./data/predictions.csv', header=['Label'], index=True, index_label='ImageId')
