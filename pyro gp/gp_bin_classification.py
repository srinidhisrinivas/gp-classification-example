# Binary Classification using Gaussian Processes in Pyro
# Dataset from https://archive.ics.uci.edu/ml/datasets/banknote+authentication

import warnings
import pandas as pd
from scipy.special import erfc
from scipy.optimize import minimize
warnings.filterwarnings("error");

import os 
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from torch.utils.data import Dataset, DataLoader

smoke_test = ('CI' in os.environ);
assert pyro.__version__.startswith('0.4.1');
pyro.enable_validation(True);
pyro.set_rng_seed(0);



class GPClassifier:
    def __init__(self, sparse=True):
        self.likelihood = gp.likelihoods.Binary();
        self.likelihood.train();
        self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        self.Xu_size=10;
        self.sparse = sparse;

    def fit(self, X, y):
        self.kernel = gp.kernels.RBF(input_dim=X.shape[1], variance=torch.tensor(1.), lengthscale=torch.tensor(1.));

        # Take inducing points at random from training set

        if self.sparse:
            indices = list(range(len(X)));
            shuffled_indices = np.random.shuffle(indices);
            split = self.Xu_size if self.Xu_size < len(X) else len(X);
            Xu = X[indices[:split]];

            self.gpr = gp.models.VariationalSparseGP(X, y, self.kernel, Xu=Xu, likelihood=self.likelihood)
        else:
            self.gpr = gp.models.VariationalGP(X, y, self.kernel, likelihood=self.likelihood)

        self.train();

    def train(self, num_steps=2500):
        gp.util.train(self.gpr, num_steps=num_steps);

    def update_post(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).float();

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y).float();

        prev_X = self.gpr.X;
        prev_y = self.gpr.y;
        assert X.shape[1] == prev_X.shape[1], 'Input shape does not match'

        new_X = torch.cat((prev_X, X), dim=0);
        new_y = torch.cat((prev_y, y), dim=0);

        gpr.set_data(new_X, new_y);

        self.train();

    def predict(self, X, y=None, acc=True):

        if acc:
            assert y is not None, 'Must supply prediction targets for accuracy'

        with torch.no_grad():
            pred_f, pred_var = self.gpr.forward(X);
            pred = self.gpr.likelihood(pred_f, pred_var);

            if acc: 
                acc_ = pred - y;
                acc_pctg = (acc_==0).sum() * 100./float(len(acc_));
                
                return pred, acc_pctg

            else:
                return pred;

if __name__ == '__main__':
    #df = pd.read_csv('data_banknote_authentication.txt', delimiter=',');
    df = pd.read_csv('haberman_data.txt', delimiter=',');

    num_features = len(df.columns)-1;

    x_vals = torch.from_numpy(df.iloc[:,0:num_features].values).float();
    y_vals = torch.from_numpy(df.iloc[:,num_features].values).float().reshape(-1,1);


    train_size = 0.8;

    np.random.seed(4);
    data = torch.cat((x_vals, y_vals),dim=1);

    indices = list(range(len(data)));
    shuffled_indices = np.random.shuffle(indices)
    split = int(math.floor(train_size * len(indices)));
    train_indices = indices[:split];
    test_indices = indices[split:];

    train_data = data[train_indices];
    train_x, train_y = train_data[:,0:num_features], train_data[:,num_features];

    test_data = data[test_indices];
    test_x, test_y = test_data[:,0:num_features], test_data[:,num_features];

    gpc = GPClassifier();

    gpc.fit(train_x, train_y);
    pred, acc = gpc.predict(test_x, test_y, acc=True)
    print('Accuracy: {0:0.3f}%'.format(acc));
