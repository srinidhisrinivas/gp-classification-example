# Program to compare different sklearn classifiers on datasets
# Source: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import pandas as pd
import numpy as np

import sys

#df = pd.read_csv('data_banknote_authentication.txt', delimiter=',');
df = pd.read_csv('haberman_data.txt', delimiter=',');

num_features = len(df.columns)-1;

train_x, test_x, train_y, test_y = train_test_split(df.values[:,0:num_features], df.values[:,num_features], train_size=0.8);
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

for name, clf in zip(names,classifiers):

	clf.fit(train_x, train_y);
	pred = clf.predict(test_x);
	acc = (pred==test_y).sum() * 100.0 / len(pred);

	print('{0}: {1:0.3f}%'.format(name, acc));