import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.autograd.variable import Variable
import numpy as np
from samplers import distribution1
import math
import matplotlib.pyplot as plt


# Using GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Discriminator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self, input_dim=2, activ_func=nn.Sigmoid()):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )

        self.out = activ_func

    def forward(self, x):
        if self.out is not None:
            return self.out(self.layers(x))
        else:
            return self.layers(x)

    def gradient_penalty(self, x, y):
        # Random weight term for interpolation between real and fake samples
        a = torch.empty_like(x).uniform_(0, 1)

        # Compute interpolated examples
        z = (a * x + ((1 - a) * y))
        z.requires_grad = True

        # Compute probability of interpolated examples
        prob_z = self.forward(z)

        # Compute gradients of probability w.r.t interpolated examples
        grads = torch.autograd.grad(
            outputs=prob_z,
            inputs=z,
            grad_outputs=torch.ones_like(prob_z),
            create_graph=True,
            only_inputs=True)[0]
        return ((torch.norm(grads, p=2, dim=1) - 1)**2).mean()

    def loss_JSD(self, x, y):
        """
        Implementation of Jensen-Shannon Divergence
        """
        return np.log(2) + (torch.mean(torch.log(self.forward(x)) / 2)) + (torch.mean(torch.log(1 - self.forward(y)) / 2))

    def loss_WD(self, x, y, lmbd=0):
        """
        Implementation of Wasserstein Distance with gradient penalty.
        """
        if lmbd is 0:
          gp = 0
        else:
          gp = self.gradient_penalty(x, y)

        return torch.mean(self.forward(x)) - torch.mean(self.forward(y)) - lmbd*gp

    def loss_func(self, x, y, metric='JSD', l=0 ):
        """
        Implementation of criterion to train the discriminator
        """
        return -self.loss_JSD(x, y) if metric == 'JSD' else -self.loss_WD(x, y, l)


def train(D, p, q, loss_metric='JSD', lmbd=0, n_epochs=50000):
    """
    Function to train the discriminator using JSD or WD
    """
    optimizer=SGD(D.parameters(), lr=1e-3)
    D.train()
    D.to(device)
    for epoch in range(n_epochs):
        x = torch.from_numpy(next(p)).float().to(device)
        y = torch.from_numpy(next(q)).float().to(device)
        optimizer.zero_grad()
        loss = D.loss_func(x, y, loss_metric, lmbd)
        loss.backward()
        optimizer.step()
        if epoch % 10000 == 0:
            print("\tEpoch", epoch, "Loss: ", -loss.item())


def predict(D, p, q, loss_metric='JSD', lmbd=0):
    """
    Function to estimate JSD or WD of trained discriminator
    """
    D.eval()
    x = torch.from_numpy(next(p)).float().to(device)
    y = torch.from_numpy(next(q)).float().to(device)
    loss = D.loss_func(x, y, loss_metric)
    return -loss.item()

if __name__ == "__main__":
    print(device)
    # Q1.3 JSD
    jsd_list = []
    wd_list = []
    phis= np.around(np.arange(-1.0, 1.0, 0.1), 1)
    for phi in phis:
        print(phi)
        dist_p = distribution1(0, 512)
        dist_q = distribution1(phi, 512)
        D = Discriminator()
        train(D, dist_p, dist_q)
        y = predict(D, dist_p, dist_q)
        print("Estimate: ", y)
        jsd_list.append(y)

    plt.scatter(phis, jsd_list)
    plt.title('{}'.format('JSD'))
    plt.ylabel('Estimated Jensen-Shannon Divergence')
    plt.xlabel('$\phi$')
    plt.savefig('JSD.png')
    plt.show()

    # Q1.3 WD
    for phi in phis:
        print(phi)
        dist_p = distribution1(0, 512)
        dist_q = distribution1(phi, 512)
        D = Discriminator(activ_func=None)
        train(D, dist_p, dist_q, loss_metric='WD', lmbd=15)
        y = predict(D, dist_p, dist_q, loss_metric='WD')
        wd_list.append(y)
        print('Estimate: ', y)

    plt.scatter(phis, wd_list)
    plt.title('{}'.format('WD'))
    plt.ylabel('Estimated Wasserstein Distance')
    plt.xlabel('$\phi$')
    plt.savefig('WD.png')
    plt.show()
