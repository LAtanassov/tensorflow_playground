import h5py
import tflearn
import pandas as pd

h5f = h5py.File('./data/redhat.h5', 'r')
xs_train = h5f['xs_train']
ys_train = h5f['ys_train']
xs_valid = h5f['xs_valid']
ys_valid = h5f['ys_valid']
xs_test = h5f['xs_test']


# Build neural network
net = tflearn.input_data(shape=[None, 52])
net = tflearn.fully_connected(net, 256)
net = tflearn.fully_connected(net, 512)
net = tflearn.fully_connected(net, 256)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(X_inputs=xs_train, Y_targets=ys_train, validation_set=(xs_valid, ys_valid), n_epoch=3, batch_size=1000, show_metric=True)

predictions = pd.DataFrame({
    'activity_id': pd.read_csv('./data/act_test.csv', usecols=['activity_id'])['activity_id'],
    'outcome': pd.DataFrame(model.predict(xs_test)).idxmax(axis=1)
})

predictions.to_csv('./data/prediction.csv', index=False)

h5f.close()