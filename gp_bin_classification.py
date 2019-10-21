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

df = pd.read_csv('data_banknote_authentication.txt', delimiter=',');

num_features = len(df.columns)-1;

x_vals = torch.from_numpy(df.iloc[:,0:num_features].values).float();
y_vals = torch.from_numpy(df.iloc[:,num_features].values).float().reshape(-1,1);

train_size = 0.8;
Xu_size=50;
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

kernel = gp.kernels.RBF(input_dim=x_vals.shape[1]);
likelihood = gp.likelihoods.Binary();
likelihood.train();
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

# Take inducing points at random from training set
indices = list(range(len(train_x)));
shuffled_indices = np.random.shuffle(indices);
split = Xu_size;
Xu = train_x[indices[:split]];

gpr = gp.models.VariationalSparseGP(train_x, train_y, kernel, Xu=Xu, likelihood=likelihood)

optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005);

losses=[]

"""
num_steps = 2500 if not smoke_test else 2;
for i in range(num_steps):
    optimizer.zero_grad();
    loss = loss_fn(gpr.model, gpr.guide)
    loss.backward()
    optimizer.step();
    losses.append(loss.item())
"""
gp.util.train(gpr, num_steps=2500);

with torch.no_grad():
    pred_f, pred_var = gpr.forward(test_x);
    pred = gpr.likelihood(pred_f, pred_var);

    #pred[pred > 0.5] = 1.;
    #pred[pred < 0.5] = 0.;
    #print(pred);
    acc = pred - test_y;
    
    acc_pctg = (acc==0).sum() * 100./len(acc);
    print('Accuracy: {0:0.3f}%'.format(acc_pctg));

