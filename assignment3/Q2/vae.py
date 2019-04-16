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
    def __init__(self, batch_size, L):
        super(VAE, self).__init__()
        self.L = L
        self.bs = batch_size

        self.encoder = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3),
                    nn.AvgPool2d(kernel_size=2, stride=2 ),
                    nn.ELU(),
                    nn.Conv2d(32, 64, kernel_size=3),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.ELU(),
                    nn.Conv2d(64, 256, kernel_size=5, stride=2),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.ELU()
                    )
        self.params = nn.Linear(256, self.L*2)
        self.linear = nn.Linear(self.L, 256)
        self.decoder = nn.Sequential(
                    nn.ELU(),
                    nn.Conv2d(256, 64, kernel_size=5, padding=4),
                    nn.ELU(),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.ELU(),
                    nn.Conv2d(64, 32, kernel_size=3, padding=2),
                    nn.ELU(),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.ELU(),
                    nn.Conv2d(32, 16, kernel_size=3, padding=2),
                    nn.ELU(),
                    nn.Conv2d(16, 1, kernel_size=3, padding=2)
                    )

    def encode(self, x):
        x = self.encoder(x)
        return self.params(x.view(-1, 256))

    def reparameterize(self, q_params):
        mu, log_sigma = q_params[:,:self.L], q_params[:,self.L:]
        sigma = torch.exp(log_sigma) + 1e-7
        e = torch.randn_like(mu, device=device)
        z = mu + sigma*e
        return z, mu, sigma

    def decode(self, z):
        z = self.linear(z)
        return self.decoder(z.view(-1, 256, 1, 1))

    def forward(self, x):
        q_params = self.encode(x)
        z, mu, sigma = self.reparameterize(q_params)
        recon_x = self.decode(z)
        return recon_x, mu, sigma


def ELBO(x, recon_x, mu, log_sigma):
    """
    Function that computes the negative ELBO
    """
    # Compute KL Divergence
    kl = 0.5 * (-1. - 2.*log_sigma + torch.exp(log_sigma)**2. + mu**2.).sum(dim=1)
    # Compute reconstruction error
    logp_z = F.binary_cross_entropy_with_logits(recon_x.view(-1, 784), x.view(-1, 784)).sum(dim=-1)
    return -(logp_z - kl).mean()

if __name__ == "__main__":

    # Load dataset
    print("Loading datasets.....")
    start_time = time.time()
    train_set = MNIST("data", split="train")
    valid_set = MNIST("data", split="valid")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
    print("DONE in {:.2f} sec".format(time.time() - start_time))

    # Set hyperparameters
    model = VAE(batch_size=1, L=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    n_epochs=20

    dataloader = {"Train": train_loader,
                  "Valid": valid_loader}

    for epoch in range(n_epochs):
        epoch += 1
        print("Epoch {} of {}".format(epoch, n_epochs))
        accuracy = {}

        for loader in ["Train", "Valid"]:
            start_time = time.time()
            train_epoch_loss = 0
            valid_epoch_loss = 0

            if loader == "Train":
                model.train()
            else:
                model.eval()

            for idx, x in enumerate(dataloader[loader]):
                optimizer.zero_grad()
                x = x.to(device)
                recon_x, mu, log_sigma = model(x)

                loss = ELBO(x, recon_x, mu, log_sigma)
                print(loss)
                if loader != "Valid":
                    loss.backward()
                    optimizer.step()

                if loader != "Valid":
                    train_epoch_loss += loss.item()
                else:
                    valid_epoch_loss += loss.item()
                if idx == 10:
                    break
        print("Train Epoch loss: {:.6f} || Valud Epoch loss: {:.6f}".format(train_epoch_loss, train_epoch_loss))
