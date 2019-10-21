# Binary Classification using Gaussian Processes in Pyro
# Data is one-dimensional, simulated from linear model through sigmoid function

import warnings
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

inv_logit = lambda x: 1.0 / (1.0 + torch.exp(-1*x));

def plot_gp(gpr, X, X_train=None, Y_train=None, samples=[], init=0, points=0, optimum=0):
    X = X.flatten()
    with torch.no_grad():
        mu, cov = gpr.forward(X, full_cov=True)
    
    uncertainty = 1.96 * cov.diag().sqrt();
    ub = mu + uncertainty;
    lb = mu-uncertainty;
    if points and optimum:
        print(points, optimum)
        plt.annotate('Points Sampled: {0}\nMaximum f(x): {1:0.2f}'.format(points, optimum), xy=(0.5,0.90), xycoords='axes fraction');


    plt.fill_between(X.numpy(), ub.numpy(), lb.numpy(), alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}', zorder=1)
    if X_train is not None:
        
        plt.scatter(X_train[0:init].numpy(), Y_train[0:init].numpy(), color='black', marker='x', zorder=5)
        plt.plot(X_train[init:len(X_train)-1].numpy(), Y_train[init:len(Y_train)-1].numpy(), 'rx')
        plt.plot(X_train[len(X_train)-1].numpy(), Y_train[len(Y_train)-1].numpy(), 'ro')
        
    plt.legend()

class GPDataset(Dataset):
    def __init__(self, size):
        self.feature_x1 = dist.Uniform(0, 100).sample(sample_shape=(size, ))
        #self.feature_x2 = dist.Uniform(0, 100).sample(sample_shape=(size, ))
        probs = inv_logit(self.feature_x1 - 50);
        self.classes = torch.tensor([dist.Bernoulli(torch.tensor([1-p])).sample() for p in probs]);

    def __len__(self):
        return self.feature_x1.shape[0];

    def __getitem__(self, idx):
        return self.feature_x1[idx], self.classes[idx];

train_size = 0.8;
np.random.seed(4);
data = GPDataset(100);
indices = list(range(len(data)));
shuffled_indices = np.random.shuffle(indices)
split = int(math.floor(train_size * len(indices)));
train_indices = indices[:split];
test_indices = indices[split:];

train_x, train_y = data[train_indices];
#train_x = torch.cat((train_x[0].reshape(-1,1), train_x[1].reshape(-1,1)), dim=1)
test_x, test_y = data[test_indices];
#test_x = torch.cat((test_x[0].reshape(-1,1), test_x[1].reshape(-1,1)), dim=1)
plt.scatter(train_x.numpy(), train_y.numpy());
#plt.show(block=False);

kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.), lengthscale=torch.tensor(5.));
likelihood = gp.likelihoods.Binary();
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

#X = torch.cat((torch.linspace(0., 100., 500).reshape(-1,1), torch.linspace(0., 100., 500).reshape(-1,1)),dim=1) 
X = torch.linspace(0., 100., 500);
Xu = torch.arange(0., 100., 10);

    #likelihood.train()
gpr = gp.models.VariationalSparseGP(train_x, train_y, kernel, Xu=Xu, likelihood=likelihood)
#gpr = gp.models.GPRegression(kernel, noise=torch.tensor([noise]))

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
gp.util.train(gpr, optimizer, num_steps=2500);

with torch.no_grad():
    pred_f = gpr.forward(test_x)[0];
    pred = inv_logit(pred_f);
    pred[pred > 0.5] = 1.;
    pred[pred < 0.5] = 0.;
    acc = pred - test_y;
    #print(acc)
    acc_pctg = (acc==0).sum() * 100/len(acc);
    print('Accuracy: {}%'.format(acc_pctg));

    Y, std = gpr.forward(X)
    plt.plot(X, inv_logit(Y), label='GP Classifer');
    plt.plot(X, inv_logit(-X+50), label='Original function');
    #plt.figure();
    #plot_gp(gpr, X);
    #plt.plot(X, X-50, color='red')

#plot_gp(gpr, , X_train=init_sample, Y_train=y_init);
plt.legend();
plt.show();


