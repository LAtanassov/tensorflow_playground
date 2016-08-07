import pandas as pd
import numpy as np
import tflearn

COLUMNS = ['Pclass', 'AgeFill', 'SibSp', 'Parch', 'Fare', 'Gender']
LABELS = ['Not_Survived', 'Survived']

df_train = pd.read_csv('./data/train.csv', header=0)
df_train['Gender'] = df_train['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_train['Not_Survived'] = df_train['Survived'].map({1: 0, 0: 1})

df_test = pd.read_csv('./data/test.csv', header=0)
df_test['Gender'] = df_train['Sex'].map({'female': 0, 'male': 1}).astype(int)

median_ages = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df_train[(df_train['Gender'] == i) & \
                              (df_train['Pclass'] == j+1)]['Age'].dropna().median()

df_train['AgeFill'] = df_train['Age']
df_test['AgeFill'] = df_test['Age']

for i in range(0, 2):
    for j in range(0, 3):
        df_train.loc[(df_train.Age.isnull()) & (df_train.Gender == i) & (df_train.Pclass == j + 1), 'AgeFill'] = median_ages[i, j]

for i in range(0, 2):
    for j in range(0, 3):
        df_test.loc[(df_test.Age.isnull()) & (df_test.Gender == i) & (df_test.Pclass == j + 1), 'AgeFill'] = median_ages[i, j]

# Build neural network
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(df_train[COLUMNS].as_matrix(), df_train[LABELS].as_matrix(), n_epoch=10, batch_size=16, show_metric=True)

df_preditions = pd.DataFrame(model.predict(df_test[COLUMNS]))
df_preditions['survived'] = np.where((df_preditions[0] < df_preditions[1]), 0, 1)
pd.concat([df_test['PassengerId'], df_preditions['survived']], axis=1).to_csv('./data/predictions.csv', header=['PassengerId', 'Survived'], index=False)