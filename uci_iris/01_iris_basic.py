# http://www.michael-remington.com/machine/learning/tensorflow/neural/networks/2016/06/25/tflearn-tutorial.html

import tempfile
import urllib
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import tflearn

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
columns = [0, 1, 2, 3]

train_file = tempfile.NamedTemporaryFile()
urllib.urlretrieve(url, train_file.name)

df_train = pd.read_csv(train_file,header=None)
y_classes = len(df_train[4].unique()) + 1
labels =  pd.get_dummies(df_train[4])


print df_train.describe()
print df_train.dtypes

scaler = preprocessing.StandardScaler()
data = scaler.fit_transform(df_train[columns])
data = pd.DataFrame(data)
print(data.describe())

X_train, X_dev, y_train, y_dev = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build neural network
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(X_train, y_train, n_epoch=10, batch_size=16, show_metric=True)