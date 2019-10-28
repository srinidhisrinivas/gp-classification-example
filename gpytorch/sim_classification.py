# Simple 1D classification using variational inference strategies in GPyTorch
# Tutorial copied from https://github.com/cornellius-gp/gpytorch/blob/master/examples/02_Simple_GP_Classification/Simple_GP_Classification.ipynb
import math
import torch
import torch.distributions as dist
import gpytorch
from matplotlib import pyplot as plt

train_x = torch.linspace(0, 1, 100)
train_y = torch.sign(torch.cos(train_x * (4 * math.pi))).add(1).div(2)
#train_y = dist.Bernoulli(1-torch.sigmoid(train_x - 50)).sample();

from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


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


# Initialize model and likelihood
model = GPClassificationModel(train_x)
likelihood = gpytorch.likelihoods.BernoulliLikelihood()

from gpytorch.mlls.variational_elbo import VariationalELBO

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
# num_data refers to the amount of training data
mll = VariationalELBO(likelihood, model, train_y.numel())

training_iter = 50
for i in range(training_iter):
    # Zero backpropped gradients from previous iteration
    optimizer.zero_grad()
    # Get predictive output
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()

# Go into eval mode
model.eval()
likelihood.eval()

with torch.no_grad():
    # Test x are regularly spaced by 0.01 0,1 inclusive
    test_x = torch.linspace(0, 1, 200)
    test_y = torch.sign(torch.cos(test_x * (4 * math.pi))).add(1).div(2)

    # Get classification predictions
    pred_f = model(test_x);
    print(pred_f);
    observed_pred = likelihood(pred_f)

    print(observed_pred.mean)

    # Initialize fig and axes for plot
    
    plt.plot(train_x.numpy(), train_y.numpy(), 'k.')
    # Get the predicted labels (probabilites of belonging to the positive class)
    # Transform these probabilities to be 0/1 labels
    pred_labels = observed_pred.mean.ge(0.5).float()
    acc_pctg = (pred_labels == test_y).sum() * 100. / float(len(test_y));
    print('Accuracy :{}'.format(acc_pctg))
    plt.plot(test_x.numpy(), pred_labels.numpy(), 'b')
    #plt.plot(test_x.numpy(), torch.sigmoid(pred_f.mean).numpy())
    #plt.plot(test_x.numpy(), torch.sigmoid(-test_x + 50).numpy())
    #plt.ylim([-1, 2])
    plt.legend(['Observed Data', 'Mean', 'Original Function'])

plt.show()