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
x_vals = torch.from_numpy(df.iloc[:,0:4].values).float();
y_vals = torch.from_numpy(df.iloc[:,4].values).float().reshape(-1,1);

train_size = 0.8;
Xu_size=0.4;
np.random.seed(4);
data = torch.cat((x_vals, y_vals),dim=1);

indices = list(range(len(data)));
shuffled_indices = np.random.shuffle(indices)
split = int(math.floor(train_size * len(indices)));
train_indices = indices[:split];
test_indices = indices[split:];

train_data = data[train_indices];
train_x, train_y = train_data[:,0:4], train_data[:,4];

test_data = data[test_indices];
test_x, test_y = test_data[:,0:4], test_data[:,4];

kernel = gp.kernels.RBF(input_dim=4);
likelihood = gp.likelihoods.Binary();
likelihood.train();
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

indices = list(range(len(train_x)));
shuffled_indices = np.random.shuffle(indices);
split = int(math.floor(Xu_size * len(indices)))
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
    pred_f = gpr.forward(test_x)[0];
    pred = torch.sigmoid(pred_f);

    pred[pred > 0.5] = 1.;
    pred[pred < 0.5] = 0.;
    acc = pred - test_y;
    
    acc_pctg = (acc==0).sum() * 100./len(acc);
    print('Accuracy: {0:0.3f}%'.format(acc_pctg));

