from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

import pandas as pd
import numpy as np

import sys

df = pd.read_csv('data_banknote_authentication.txt', delimiter=',');

train_x, test_x, train_y, test_y = train_test_split(df.values[:,0:4], df.values[:,4], train_size=0.05);

gpc = GaussianProcessClassifier().fit(train_x, train_y);

pred = gpc.predict(test_x);
acc = (pred==test_y).sum() * 100.0 / len(pred);

print(acc);