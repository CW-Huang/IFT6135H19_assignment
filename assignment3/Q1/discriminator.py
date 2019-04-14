import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torch.autograd import grad as torch_grad
import numpy as np
import samplers

import matplotlib.pyplot as plt


# Using GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Discriminator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], out_dim),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.layers(x), self.layers(y)

def gradient_penalty(D, x, y, bs):
    # Sample from Uniform distribution
    dist_u = iter(distribution1(0))
    a = torch.from_numpy(next(u)).float()
    a.to(device)

    # Compute interpolated examples
    z = a*x + (1-a)*y
    z = Variable(z, requires_grad=True)
    z.to(device)

    # Compute probability of interpolated examples
    prob_z = D(z)

    # Compute gradients of probability w.r.t interpolated examples
    grads_z = torch_grad(outputs=prob_z, inputs=z, grad_outputs=torch.ones(prob_z.size()).cuda() if torch.cuda.is_available() else  torch.ones(prob_z.size()), create_graph=True, retain_graph=True)[0]

    # Compute norm of gradients
    grads_norm = torch.sqrt(torch.sum(grads_z ** 2, dim=1) + 1e-12)

    return ((grads_norm - 1) ** 2).mean()


def loss_JSD(x, y):
    """
    Implementation of Jensen-Shannon Divergence
    """
    return -(torch.log(torch.Tensor([2])) + 0.5*torch.mean(torch.log(x)) + 0.5*torch.mean(torch.log(1 - y)))

def loss_WD(x, y, l, penalty):
    """
    Implementation of Wasserstein Distance with gradient penalty.
    """
    penality = gradient_penalty(D, x, y, bs)
    return -(torch.mean(x) - torch.mean(y) - l*penality)

def train(D, p, q, loss_metric='JSD', lmbd=35, n_epochs=2000):
    """
    Function to train the discriminator
    """
    optimizer=SGD(D.parameters(), lr=1e-3)
    D.train()
    running_loss = 0

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        sample_x = torch.from_numpy(next(p)).float()
        sample_x.to(device)
        sample_y = torch.from_numpy(next(q)).float()
        sample_y.to(device)
        x, y = D(sample_x, sample_y)

        if loss_metric == 'JSD':
            loss = loss_JSD(x, y)
        elif loss_metric == 'WD':
            gp = gradient_penalty(D, x, y, 512)
            loss = loss_WD(x, y, lmbd, gp)

        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print("\tEpoch", epoch, "Loss: ", loss.data.numpy())

    return running_loss / n_epochs

def estimate(D, p, q, loss_metric='JSD'):
    """
    Function to estimate JSD or WD of trained discriminator
    """
    D.eval()

    sample_x = torch.from_numpy(next(p)).float()
    sample_x.to(device)
    sample_y = torch.from_numpy(next(q)).float()
    sample_y.to(device)
    x, y = D(sample_x, sample_y)

    if loss_metric == 'JSD':
        loss = loss_JSD(x, y)
    elif loss_metric == 'WD':
        gp = gradient_penalty(D, x, y, 512)
        loss = loss_WD(x, y, lmbd, gp)

    return loss


if __name__ == "__main__":

    # Q1.3 JSD
    y_axis = []
    x_axis = []

    for phi in np.around(np.arange(-1.0, 1.0, 0.1), 1):
        dist_p = samplers.distribution1(0, 512)
        dist_q = samplers.distribution1(phi, 512)
        D = Discriminator(2, [84,84], 1)
        loss = train(D, dist_p, dist_q)
        y = estimate(D, dist_p, dist_q)
        y_axis.append(y)
        x_axis.append(phi)

    plt.plot(x_axis, y_axis, '.')
    plt.xlim(-1, 1)
    plt.title('{}'.format('JSD'))
    plt.savefig('test.png')
    plt.show()

    # Q1.3 WD
    y_axis = []
    x_axis = []

    for phi in np.around(np.arange(-1.0, 1.0, 0.1), 1):
        dist_p = samplers.distribution1(0, 512)
        dist_q = samplers.distribution1(phi, 512)
        D = Discriminator(2, [84,84], 1, loss_metric='WD')
        loss = train(D, dist_p, dist_q)
        y = estimate(D, dist_p, dist_q)
        y_axis.append(y)
        x_axis.append(phi)

    plt.plot(x_axis, y_axis, '.')
    plt.xlim(-1, 1)
    plt.title('{}'.format('JSD'))
    plt.savefig('test.png')
    plt.show()
