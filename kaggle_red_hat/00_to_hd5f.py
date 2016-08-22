import numpy as np
import pandas as pd


train = pd.read_csv('./data/act_train.csv', parse_dates=['date'])
test = pd.read_csv('./data/act_test.csv', parse_dates=['date'])
ppl = pd.read_csv('./data/people.csv', parse_dates=['date'])

df_train = pd.merge(train, ppl, on='people_id')
df_test = pd.merge(test, ppl, on='people_id')


# Save the test IDs for Kaggle submission
del train, test, ppl

df_labels = pd.get_dummies(df_train.pop('outcome'))
df_train = df_train.drop(['people_id', 'activity_id'], axis=1)
df_test = df_test.drop(['people_id', 'activity_id'], axis=1)

for col in df_train.columns:

    if (df_train[col].dtype == 'object'):
        df_train[col] = df_train[col].fillna('type 0')
        df_train[col] = df_train[col].apply(lambda x: x.split(' ')[1])
        df_train[col] = pd.to_numeric(df_train[col]).astype(float)

        df_test[col] = df_test[col].fillna('type 0')
        df_test[col] = df_test[col].apply(lambda x: x.split(' ')[1])
        df_test[col] = pd.to_numeric(df_test[col]).astype(float)
    elif (df_train[col].dtype == 'bool'):
        df_train[col] =  pd.to_numeric(df_train[col]).astype(float)
        df_test[col] =  pd.to_numeric(df_test[col]).astype(float)
    elif (df_train[col].dtype == 'datetime64[ns]'):

        df_train['year'] = df_train[col].dt.year
        df_test['year'] = df_test[col].dt.year

        df_train[col] =  df_train[col].dt.dayofweek
        df_test[col] =  df_test[col].dt.dayofweek


msk = np.random.rand(len(df_train)) < 0.8
xs_train = df_train[msk].as_matrix()
ys_train = df_labels[msk].as_matrix()

xs_valid = df_train[~msk].as_matrix()
ys_valid = df_labels[~msk].as_matrix()

import h5py
h5f = h5py.File('./data/redhat.h5', 'w')
h5f.create_dataset('xs_train', data=df_train[msk].as_matrix())
h5f.create_dataset('ys_train', data=df_labels[msk].as_matrix())
h5f.create_dataset('xs_valid', data=df_train[~msk].as_matrix())
h5f.create_dataset('ys_valid', data=df_labels[~msk].as_matrix())
h5f.create_dataset('xs_test', data=df_test.as_matrix())
h5f.close()