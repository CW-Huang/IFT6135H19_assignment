import os
import time
import json
import hashlib
import argparse
import time

import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn, optim
from torch import autograd
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from dataset import MNIST
from dataloader import get_data_loader

# Using GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class VAE(nn.Module):
    def __init__(self, L):
        super(VAE, self).__init__()
        self.L = L

        self.encoder = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(32, 64, kernel_size=3),
                    nn.ELU(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 256, kernel_size=5),
                    nn.ELU()
                    )
        self.params = nn.Linear(256, self.L*2)
        self.linear = nn.Linear(self.L, 256)
        self.decoder = nn.Sequential(
                    nn.ELU(),
                    nn.Conv2d(256, 64, kernel_size=5, padding=4),
                    nn.ELU(),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(64, 32, kernel_size=3, padding=2),
                    nn.ELU(),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(32, 16, kernel_size=3, padding=2),
                    nn.ELU(),
                    nn.Conv2d(16, 1, kernel_size=3, padding=2),
                    nn.Sigmoid()
                    )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), 256)
        return self.params(x)

    def reparameterize(self, q_params):
        mu, log_sigma = q_params[:,:self.L], q_params[:,self.L:]
        sigma = 1e-10 + torch.sqrt(torch.exp(log_sigma))

        e = torch.randn(q_params.size(0), self.L, device=device)
        z = mu + sigma * e
        return z, mu, log_sigma, sigma

    def decode(self, z):
        z = self.linear(z)
        z = self.decoder(z.view(z.size(0), 256, 1, 1))
        return z

    def forward(self, x):
        q_params = self.encode(x)
        z, mu, log_sigma, sigma = self.reparameterize(q_params)
        recon_x = self.decode(z)
        return recon_x, mu, log_sigma, sigma


def ELBO(x, recon_x, mu, sigma):
    """
    Function that computes the negative ELBO
    """
    # Compute KL Divergence
    kld = 0.5 * torch.sum(-1 - torch.log(sigma ** 2) + mu ** 2 + sigma ** 2 )
    # Compute reconstruction error
    logpx_z = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return logpx_z + kld

if __name__ == "__main__":

    # Load dataset
    print("Loading datasets.....")
    start_time = time.time()
    train_set = MNIST("data", split="train")
    valid_set = MNIST("data", split="valid")
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False)
    print("DONE in {:.2f} sec".format(time.time() - start_time))

    # Set hyperparameters
    model = VAE(L=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3*1e-4, eps=1e-4)
    n_epochs=20

    dataloader = {"Train": train_loader,
                  "Valid": valid_loader}

    for epoch in range(n_epochs):
        epoch += 1
        # print("Epoch {} of {}".format(epoch, n_epochs))
        train_epoch_loss = 0
        valid_epoch_loss = 0

        for loader in ["Train", "Valid"]:
            if loader != "Valid":
                model.train()
            else:
                model.eval()

            for idx, x in enumerate(dataloader[loader], 1):
                optimizer.zero_grad()
                x = x.to(device)
                recon_x, mu, log_sigma, sigma = model(x)

                loss = ELBO(x, recon_x, mu, sigma)

                if loader != "Valid":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_epoch_loss += loss.item()
                else:
                    valid_epoch_loss += loss.item()
                if idx % 100 == 0 and loader == "Train":
                    print('\tTrain ELBO: {:.6f}'.format(-loss.item() / x.size(0)))

            if loader != "Valid":
                print("Epoch {} - Train ELBO: {:.6f}".format(epoch, -train_epoch_loss / len(dataloader[loader].dataset)))
            else:
                print("Epoch {} - Valid ELBO: {:.6f}".format(epoch, -valid_epoch_loss / len(dataloader[loader].dataset)))
