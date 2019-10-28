# Binary Classification using Gaussian Processes in GPyTorch
# Dataset is a simulated 2D sinusoid 
from gpytorch.mlls.variational_elbo import VariationalELBO
import math
import torch
import torch.distributions as dist
import gpytorch
from matplotlib import pyplot as plt
import os

import pandas as pd
import numpy as np
import sys
import time

from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

torch.manual_seed(2);

class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

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
    def __init__(self, sparse=False):
        # Initialize model and likelihood
        
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        self.likelihood.train();
        self.sparse = sparse;

    def fit(self, X, y):
        
        if self.sparse:
            perm = torch.randperm(X.size(0))
            idx = perm[:50] 
            self.inducing_points = X[idx];
            self.inducing_points.sort();
            class_model = VSGPClassificationModel;
        else:
            self.inducing_points = X;
            class_model = GPClassificationModel;

        plt.scatter(self.inducing_points[:,0].numpy(), self.inducing_points[:,1].numpy())
        plt.xlim([-1,1])
        plt.show()
        self.X = X;
        self.y = y;
        self.model = class_model(self.inducing_points);
        # Use the adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        # num_data refers to the amount of training data
        self.mll = VariationalELBO(self.likelihood, self.model, self.y.numel())

        self.train();

    def train(self, num_steps=50):

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

    def update_post(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).float();

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y).float();

        prev_X = self.X;
        prev_y = self.y;
        assert X.shape[1] == prev_X.shape[1], 'Input shape does not match'

        self.X = torch.cat((prev_X, X), dim=0);
        self.y = torch.cat((prev_y, y), dim=0);

        gpr.set_data(new_X, new_y);

        self.train();

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

            print(observed_pred.mean)

            # Initialize fig and axes for plot
            
            #plt.plot(train_x.numpy(), train_y.numpy(), 'k.')
            # Get the predicted labels (probabilites of belonging to the positive class)
            # Transform these probabilities to be 0/1 labels
            pred_labels = observed_pred.mean.ge(0.5).float()
            #plt.plot(test_x.numpy(), pred_labels.numpy(), 'b')
            #plt.plot(test_x.numpy(), torch.sigmoid(pred_f.mean).numpy())
            #plt.plot(test_x.numpy(), torch.sigmoid(-test_x + 50).numpy())
            #plt.ylim([-1, 2])
            #plt.legend(['Observed Data', 'Mean', 'Original Function'])

            if acc:
                acc_pctg = (pred_labels == y).sum() * 100 / len(y);
                return pred_labels, acc_pctg
            else:
                return pred_labels

if __name__ == '__main__':
    
    
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
    
    gpc = GPClassifier(True);

    start = time.time()
    gpc.fit(train_x, train_y);
    end = time.time()

    print('Fitting: {}s'.format(end-start));

    start = time.time()
    pred_labels, acc = gpc.predict(test_x, test_y, acc=True)
    end = time.time()

    print('Predicting: {}s'.format(end-start));

    print('Accuracy: {0:0.3f}%'.format(acc));

    fig, ax = plt.subplots(1,2, figsize=(12, 4))
    color1 = []
    color2 = [];
    for i in range(len(pred_labels)):
        if test_y[i] == 1:
            color2.append('y');
        else:
            color2.append('r');

        if pred_labels[i] == 1:
            color1.append('y')
        else:
            color1.append('r')
        
    # Plot data a scatter plot
    ax[0].scatter(test_x[:, 0].cpu(), test_x[:, 1].cpu(), color=color1, s=1)
    ax[1].scatter(test_x[:, 0].cpu(), test_x[:, 1].cpu(), color=color2, s=1)
    plt.show();
