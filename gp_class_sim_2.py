# Binary Classification using Gaussian Processes in Pyro
# Dataset simulated from sinusoidal functions

from gp_bin_classification import GPClassifier

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
def plot_gp(gpr, X, X_train=None, Y_train=None, samples=[], init=0, points=0, optimum=0):
    with torch.no_grad():
        mu, cov = gpr.forward(X, full_cov=True)
    
    uncertainty = 1.96 * cov.diag().sqrt();
    ub = mu + uncertainty;
    lb = mu-uncertainty;
    if points and optimum:
        print(points, optimum)
        plt.annotate('Points Sampled: {0}\nMaximum f(x): {1:0.2f}'.format(points, optimum), xy=(0.5,0.90), xycoords='axes fraction');


    plt.fill_between(X.flatten().numpy(), ub.numpy(), lb.numpy(), alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}', zorder=1)
    if X_train is not None:
        
        plt.scatter(X_train[0:init].numpy(), Y_train[0:init].numpy(), color='black', marker='x', zorder=5)
        plt.plot(X_train[init:len(X_train)-1].numpy(), Y_train[init:len(Y_train)-1].numpy(), 'rx')
        plt.plot(X_train[len(X_train)-1].numpy(), Y_train[len(Y_train)-1].numpy(), 'ro')
        
    plt.legend()
size=500;

#data_x = dist.Uniform(0, 100).sample(sample_shape=(size, )).reshape(-1,1);
#probs = torch.sigmoid(data_x - 50);

#data_y = torch.tensor([dist.Bernoulli(torch.tensor([1-p])).sample() for p in probs]).reshape(-1,1);

data_x = dist.Uniform(-5, 5).sample(sample_shape=(size,)).reshape(-1,1);
data_y = torch.sign(torch.sin(data_x));
data_y[data_y == -1] = 0;

plt.scatter(data_x.numpy(), data_y.numpy())
plt.show()

num_features = data_x.shape[1];

train_size = 0.8;

np.random.seed(4);
data = torch.cat((data_x, data_y),dim=1);

indices = list(range(len(data)));
shuffled_indices = np.random.shuffle(indices)
split = int(math.floor(train_size * len(indices)));
train_indices = indices[:split];
test_indices = indices[split:];

train_data = data[train_indices];
train_x, train_y = train_data[:,0:num_features], train_data[:,num_features];

test_data = data[test_indices];
test_x, test_y = test_data[:,0:num_features], test_data[:,num_features];

gpc = GPClassifier(sparse=True);

gpc.fit(train_x, train_y);
pred, acc = gpc.predict(test_x, test_y, acc=True)
print('Accuracy: {0:0.3f}%'.format(acc));

X = torch.linspace(0., 100., 500).reshape(-1,1);
with torch.no_grad():
    Y, std = gpc.gpr.forward(X);

    plt.scatter(train_x.numpy(), train_y.numpy())
    plt.plot(X, torch.sigmoid(Y), label='GP Classifer');
    plt.plot(X, torch.sigmoid(-X + 50), label='Original function');
    plt.figure();
    print(gpc.gpr.X.shape, X.shape);
    plot_gp(gpc.gpr, X);

plt.show()