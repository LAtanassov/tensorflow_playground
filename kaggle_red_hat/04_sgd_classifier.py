
import h5py
import numpy as np
import pandas as pd


h5f = h5py.File('./data/redhat.h5', 'r')
xs_train = h5f['xs_train']
ys_train = h5f['ys_train']
xs_valid = h5f['xs_valid']
ys_valid = h5f['ys_valid']
xs_test = h5f['xs_test']

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,BaggingClassifier
from sklearn.linear_model import SGDClassifier

classifier = [
    #AdaBoostClassifier(n_estimators=10),
    #GradientBoostingClassifier(n_estimators=30),
    BaggingClassifier(n_estimators=30)
    #SGDClassifier()
    #RandomForestClassifier(bootstrap=False, n_estimators=50)
]


for clf in classifier:
    clf.fit(xs_train, np.argmax(ys_train, axis=1) + 1)
    print clf
    print clf.score(xs_valid, np.argmax(ys_valid, axis=1) + 1)

#predictions = pd.DataFrame({'activity_id': pd.read_csv('./data/act_test.csv', usecols=['activity_id'])['activity_id'], 'outcome': clf.predict(xs_test) - 1 })
#predictions.to_csv('./data/prediction.csv', index=False)
h5f.close()