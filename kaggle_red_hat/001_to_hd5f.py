import numpy as np
import pandas as pd

train = pd.read_csv('./data/act_train.csv', parse_dates=['date'])
ppl = pd.read_csv('./data/people.csv', parse_dates=['date'])

df_train = pd.merge(train, ppl, on='people_id')
del train, ppl

df_labels = pd.get_dummies(df_train.pop('outcome'))
df_train = df_train.drop(['activity_id'], axis=1)

df_train['people_id'] = df_train['people_id'].apply(lambda x: x.replace('_',' ').split(' ')[1])
df_train['people_id'] = pd.to_numeric(df_train['people_id']).astype(int)

print df_train['people_id'].unique().shape
print len(df_train.groupby('people_id'))
