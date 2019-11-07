# Binary Classification using Gaussian Processes in GPyTorch
# Dataset is a simulated 2D sinusoid 
# Uses discrete grid to sample points in 2D space

from gpytorch.mlls.variational_elbo import VariationalELBO
import math
import torch
import torch.distributions as dist
import gpytorch
from matplotlib import pyplot as plt
import os
from scipy.optimize import minimize

import pandas as pd
import numpy as np
import seaborn as sns
import sys
import time

from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

torch.manual_seed(4);

class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x, lengthscale=None):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        if lengthscale is not None:
        	self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale=lengthscale))
        else:
        	self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def set_data(self, new_X):
        self.__init__(new_X);

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class VSGPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution, learn_inducing_locations=True)
        super(VSGPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class GPClassifier:
    def __init__(self, sparse=False, n_dims=1):
        # Initialize model and likelihood
        
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        self.likelihood.train();
        self.sparse = sparse;
        self.n_dims = n_dims
        self.X = None;
        self.y = None;

        self.training_time = 0.0;

    def fit(self, X, y, lengthscale=None):
        
        if self.sparse:
            perm = torch.randperm(X.size(0))
            idx = perm[:50]
            self.inducing_points = X[idx];
            self.inducing_points.sort();
            class_model = VSGPClassificationModel;
        else:
            self.inducing_points = X;
            class_model = GPClassificationModel;

        self.X = X;
        self.y = y;

        self.model = class_model(self.inducing_points, lengthscale);
        # Use the adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        # num_data refers to the amount of training data
        self.mll = VariationalELBO(self.likelihood, self.model, self.y.numel())

        self.train();

    def train(self, num_steps=50):

        start = time.time();
        for i in range(num_steps):
            # Zero backpropped gradients from previous iteration
            self.optimizer.zero_grad()
            # Get predictive output
            output = self.model(self.X)
            # Calc loss and backprop gradients
            loss = -self.mll(output, self.y)
            loss.backward()
            #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
            self.optimizer.step()

        end = time.time()
        self.training_time += end - start;

    def update_post(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).float();

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y).float();

        if self.X is not None:
            #print(X.shape, self.X.shape)
            assert X.shape[1] == self.X.shape[1], 'Input shape does not match'

        else:
            self.fit(X, y);
            return 0;

        lengthscale = self.model.covar_module.base_kernel.lengthscale.item()    
        self.fit(torch.cat((self.X, X), dim=0), torch.cat((self.y, y), dim=0), lengthscale=lengthscale);

    def forward(self, X):
        with torch.no_grad():
            pred_f = self.model(X);

            return pred_f;

    def predict(self, X, y=None, acc=True):

        if acc:
            assert y is not None, 'target labels required to calculate accuracy'

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            # Test x are regularly spaced by 0.01 0,1 inclusive
            test_x = X
            # Get classification predictions
            pred_f = self.model(test_x);
            
            observed_pred = self.likelihood(pred_f)

            # Get the predicted labels (probabilites of belonging to the positive class)
            # Transform these probabilities to be 0/1 labels
            pred_labels = observed_pred.mean.ge(0.5).float()

            if acc:
                acc_pctg = (pred_labels == y).sum() * 100 / len(y);
                return pred_labels, acc_pctg
            else:
                return pred_labels

def RandSample(gpc, X):
    return torch.randperm(X.size(0))[0]

def plot_cmap(fig, gpc, X, axpoints, n, cmap):
    with torch.no_grad():
        post_f = gpc.forward(X);
        mean, var = post_f.mean, post_f.variance
        pcm = plt.pcolormesh(axpoints, axpoints, var.detach().numpy().reshape(n,n), cmap=cmap)#, vmin=0., vmax=0.5);
        fig.colorbar(pcm);

def plot_points(X, selected):
    train_x = X;
    plt.scatter(train_x[:,0].numpy(), train_x[:,1].numpy(), color='yellow', s=0.8, alpha=0.5);
    plt.scatter(selected[:,0].numpy(), selected[:,1].numpy(), color='black', s=7, alpha=0.8);
    plt.scatter(selected[-1][0], selected[-1][1], color='blue', s=10, alpha=1);

def UCB(gpc, X):
    with torch.no_grad():
        post_f = gpc.forward(X)

        idx = torch.argmax(post_f.variance);

        return idx;

if __name__ == '__main__':
    cmap = plt.get_cmap('YlOrRd');
    fig = plt.figure(figsize=(12,6));

    num_steps = 30;

    n = 20
    train_x = torch.zeros(n ** 2, 2)
    train_x[:, 0].copy_(torch.linspace(-1, 1, n).repeat(n))
    train_x[:, 1].copy_(torch.linspace(-1, 1, n).unsqueeze(1).repeat(1, n).view(-1))
    train_y = torch.sign(torch.sin(train_x[:, 0] + 2*train_x[:, 1] * math.pi)).add(1).div(2);

    n = 100
    test_x = torch.zeros(n ** 2, 2)
    test_x[:, 0].copy_(torch.linspace(-1, 1, n).repeat(n))
    test_x[:, 1].copy_(torch.linspace(-1, 1, n).unsqueeze(1).repeat(1, n).view(-1))
    test_y = torch.sign(torch.sin(test_x[:, 0] + 2*test_x[:, 1] * math.pi)).add(1).div(2);
    
    
    
    gpc_list = [GPClassifier(sparse=False, n_dims=2), GPClassifier(sparse=False, n_dims=2)];
    gpc_names = ['UCB', 'Random Sample'];
    ac_funcs = [UCB, RandSample]
    next_idx = RandSample(gpc_list[0], train_x);

    next_x = [train_x[next_idx].unsqueeze(0), train_x[next_idx].unsqueeze(0)];
    next_y = [train_y[next_idx].unsqueeze(0), train_y[next_idx].unsqueeze(0)];
    
    selected = next_x.copy();

    pred_labels = [None, None]

    accs = [[],[]]

    for i in range(num_steps):
        for j, gpc in enumerate(gpc_list):     
            if i != 0:
                selected[j] = torch.cat((selected[j], next_x[j]), dim=0);

            gpc.update_post(next_x[j], next_y[j])
        
            next_idx = ac_funcs[j](gpc, train_x);
            
            next_x[j], next_y[j] = train_x[next_idx].unsqueeze(0), train_y[next_idx].unsqueeze(0);

            if (i+1) % 5 == 0:
                pred_labels[j], acc = gpc.predict(test_x, test_y, acc=True)
                #print('{0}: Accuracy on test set, iteration {1}: {2:0.3f}%'.format(gpc_names[j],i+1, acc));
                accs[j].append(acc);
        
        plt.pause(1)
        plt.clf();
        plt.annotate('Points sampled: {}'.format(i + 1), xy=(0.5,0.90))#, xycoords='axes fraction')
        for j, gpc in enumerate(gpc_list):
            plt.subplot(1,2,j+1);

            plot_cmap(fig, gpc, test_x, torch.linspace(-1,1,n), n, cmap);
            plot_points(train_x, selected[j])
            
            plt.show(block=False);
        
    print('Time spent training gpc1: {0:0.3f} s'.format(gpc_list[0].training_time));
    print('Time spent training gpc2: {0:0.3f} s'.format(gpc_list[1].training_time));

    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    ax = ax.flatten();
    for j in range(len(gpc_list)):
        ax[j].plot((np.array(list(range(len(accs[j])))) + 1)*5, accs[j]);
        ax[j].set_xlabel('Trials')
        ax[j].set_ylabel('Accuracy \% on validation set');
        ax[j].set_title(gpc_names[j])
        ax[j].set_ylim([0,110]);
        ax[j].axhline(y=100, linestyle='--');


    fig, ax = plt.subplots(2,2, figsize=(12, 8))
    ax=ax.flatten();
    color1 = []
    color2 = [];
    color3 = []
    for i in range(len(pred_labels[0])):
        if test_y[i] == 1:
            color3.append('y');
        else:
            color3.append('r');

        if pred_labels[0][i] == 1:
            color1.append('y')
        else:
            color1.append('r')

        if pred_labels[1][i] == 1:
            color2.append('y')
        else:
            color2.append('r')
    
    
    # Plot data a scatter plot
    ax[0].scatter(test_x[:, 0].cpu(), test_x[:, 1].cpu(), color=color1, s=1)
    ax[2].scatter(test_x[:, 0].cpu(), test_x[:, 1].cpu(), color=color2, s=1)
    ax[1].scatter(test_x[:, 0].cpu(), test_x[:, 1].cpu(), color=color3, s=1)
    ax[3].scatter(test_x[:, 0].cpu(), test_x[:, 1].cpu(), color=color3, s=1)

    plt.show();
