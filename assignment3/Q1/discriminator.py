import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torch.autograd import torch_grad

import samplers

class Discriminator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self, input_dim, hidden_dim, out_dim, dropout):
        super(DiscriminatorNet, self).__init__()
        n_features = 1
        n_out = 1

        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, hidden_dim[0]),
            nn.ReLU()
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        return x

def JSD(x, y):
    """
    Implementation of pairwise Jensen-Shannon Divergence
    """
    return np.log(2) + (0.5*torch.mean(torch.log(x)) + 0.5*torch.mean(torch.log(1 - y)))

def WD(x, y, bs, l):
    """
    Implementation of Wasserstein Distance with gradient penalty.
    """
    penality = gradient_penalty(x, y, bs)
    return -(torch.mean(x) - torch.mean(y) - l*penality)

def gradient_penalty(discriminator x, y, bs):
    # Sample from Uniform distribution
    u = samplers.distribution2(batch_size=bs)
    a = torch.from_numpy(next(u)).float()
    a.to(device)

    # Compute interpolated examples
    z = a*x + (1-a)*y
    z = Variable(z, requires_grad=True)
    z.to(device)

    # Compute probability of interpolated examples
    prob_z = discriminator(z)

    # Compute gradients of probability w.r.t interpolated examples
    grads_z = torch_grad(outputs=prob_z, inputs=z, grad_outputs=torch.ones(prob_z.size()).cuda() if torch.cuda.is_available() else  torch.ones(prob_z.size()), create_graph=True, retain_graph=True)[0]

    # Compute norm of gradients
    grads_norm = torch.sqrt(torch.sum(grads_z ** 2, dim=1) + 1e-12)

    return ((grads_norm - 1) ** 2).mean())

def train(p, q, D, loss_func='JSD', l=15, optimizer=SGD(lr=1e-3), n_epochs=50):
    """
    Function to train the discriminator
    """
    optimizer.zero_grad()

    for epoch in range(n_epochs):
        sample_x = torch.from_numpy(next(p)).float()
        sample_x.to(device)
        sample_y = torch.from_numpy(next(q)).float()
        sample_y.to(device)
        x = D(sample_x)
        y = D(sample_y)

        if loss_func == 'JSD':
            x = F.sigmoid(x)
            y = F.sigmoid(y)
            loss = JSD(x, y)
        elif loss_func == 'WD':
            loss = WD(D, x, y, bs, l)
        loss.backward()
        optimizer.step()

    return D
