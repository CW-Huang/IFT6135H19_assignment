import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torch.autograd import grad as torch_grad
import numpy as np
import samplers
import math
import matplotlib.pyplot as plt


# Using GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Discriminator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self, input_dim, activation_func=None):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
            )

        self.out = activation_func

    def forward(self, x):
        if self.out is not None:
            return self.out(self.layers(x))
        else:
            return self.layers(x)

def gradient_penalty(D, x, y, bs):
    # Sample from Uniform distribution
    a = torch.rand(x.size()[0],1)
    a.expand_as(x)
    a.to(device)

    # Compute interpolated examples
    z = a*x + (1-a)*y
    z = Variable(z, requires_grad=True)

    # Compute probability of interpolated examples
    prob_z = D(z)

    # Compute gradients of probability w.r.t interpolated examples
    grads_z = torch_grad(outputs=prob_z, inputs=z, grad_outputs=torch.ones(prob_z.size()), create_graph=True, retain_graph=True)[0]

    # Compute norm of gradients
    grads_norm = torch.sqrt(torch.sum(grads_z ** 2, dim=1) + 1e-12)

    return ((grads_norm - 1) ** 2).mean()


def loss_JSD(x, y):
    """
    Implementation of Jensen-Shannon Divergence
    """
    return -(math.log(2.0) + torch.mean(torch.log(x)) / 2 + 0.5*torch.mean(torch.log(1 - y)) / 2)

def loss_WD(x, y, l, gp):
    """
    Implementation of Wasserstein Distance with gradient penalty.
    """
    return -(torch.mean(x) - torch.mean(y) + l*gp)

def train(D, p, q, loss_metric='JSD', lmbd=35, n_epochs=10000):
    """
    Function to train the discriminator
    """
    optimizer=SGD(D.parameters(), lr=1e-3)
    D.train()

    for epoch in range(n_epochs):
        sample_x, sample_y = torch.from_numpy(next(p)).float(), torch.from_numpy(next(q)).float()
        sample_x.to(device)
        sample_y.to(device)
        x, y = D(sample_x), D(sample_y)

        if loss_metric == 'JSD':
            loss = loss_JSD(x, y)
        elif loss_metric == 'WD':
            gp = gradient_penalty(D, x, y, 512)
            loss = loss_WD(x, y, lmbd, gp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5000 == 0:
            print("\tEpoch", epoch, "Loss: ", -loss.data.numpy())


def estimate(D, p, q, loss_metric='JSD', lmbd=35):
    """
    Function to estimate JSD or WD of trained discriminator
    """
    D.eval()
    sample_x, sample_y = torch.from_numpy(next(p)).float(), torch.from_numpy(next(q)).float()
    sample_x.to(device)
    sample_y.to(device)
    x, y = D(sample_x), D(sample_y)

    if loss_metric == 'JSD':
        loss = loss_JSD(x, y)
    elif loss_metric == 'WD':
        gp = gradient_penalty(D, x, y, 512)
        loss = loss_WD(x, y, lmbd, gp)

    return -loss.data.cpu().numpy()



if __name__ == "__main__":

    # Q1.3 JSD
    jsd_list = []
    wd_list = []
    phis= np.around(np.arange(-1.0, 1.0, 0.1), 1)
    dist_p = samplers.distribution1(0, 512)
    for phi in phis:
        print(phi)
        dist_q = samplers.distribution1(phi, 512)
        D = Discriminator(2, activation_func=nn.Sigmoid())
        train(D, dist_p, dist_q)
        y = estimate(D, dist_p, dist_q)
        print("Estimate: ",  y)
        jsd_list.append(y)
        phis.append(phi)

    plt.plot(phis, jsd_list, '.')
    plt.xlim(-1, 1)
    plt.title('{}'.format('JSD'))
    plt.savefig('test.png')
    plt.show()

    # Q1.3 WD

    for phi in phis:
        dist_p = samplers.distribution1(0, 512)
        dist_q = samplers.distribution1(phi, 512)
        D = Discriminator(2, [84,84, 84], 1)
        train(D, dist_p, dist_q, loss_metric='WD')
        y = estimate(D, dist_p, dist_q, loss_metric='WD')
        wd_list.append(y)
        phis.append(phi)

    plt.plot(phis, wd_list, '.')
    plt.xlim(-1, 1)
    plt.title('{}'.format('JSD'))
    plt.savefig('test.png')
    plt.show()
