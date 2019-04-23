import os
import time
import json
import hashlib
import argparse
import time
import numpy as np
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

# Ignore pytorch depracation warnings
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

# Using GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

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
        q_params.to(device)
        z, mu, log_sigma, sigma = self.reparameterize(q_params)
        recon_x = self.decode(z)
        return recon_x, mu, log_sigma, sigma


def ELBO(x, recon_x, mu, sigma):
    # Compute KL Divergence
    kld = 0.5 * torch.sum(-1 - torch.log(sigma ** 2) + mu ** 2 + sigma ** 2 )
    # Compute reconstruction error
    logpx_z = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return logpx_z + kld


def train(model, train_loader, valid_loader, n_epochs=20,):
    dataloader = {"Train": train_loader,
                  "Valid": valid_loader}
    optimizer = optim.Adam(model.parameters(), lr=3*1e-4, eps=1e-4)

    for epoch in range(n_epochs):
        epoch += 1
        train_loss = 0
        valid_loss = 0

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
                    train_loss += loss.item()
                else:
                    valid_loss += loss.item()
                if idx % 100 == 0 and loader == "Train":
                    print('\tTrain ELBO: {:.6f}'.format(-loss.item() / x.size(0)))

            if loader != "Valid":
                train_epoch_loss = - train_loss /len(dataloader[loader].dataset)
                print("Epoch {} - Train ELBO: {:.6f}".format(epoch, train_epoch_loss))
            else:
                valid_epoch_loss = - valid_loss / len(dataloader[loader].dataset)
                print("Epoch {} - Valid ELBO: {:.6f}".format(epoch, -valid_epoch_loss))
    return -train_epoch_loss, -valid_epoch_loss


def importance_sampling(model, x, K=200, L=100):
    # mini-batch size
    M = x.size(0)
    with torch.no_grad():
        recon_x, mu, log_sigma, sigma = model(x)
        recon_x = recon_x.to(device)
        mu = mu.to(device)
        sigma = sigma.to(device)

        logpx = torch.FloatTensor(K, M).to(device)
        for i in range(K):
            e = torch.rand(M, L).to(device)
            z = mu + sigma * e
            recon_x = model.decode(z)

            # p(x|z)
            recon_x = recon_x.view(M, -1).to(device)
            x = x.view(M, -1).to(device)
            logpxz = F.binary_cross_entropy(recon_x, x, reduction='none'
                ).sum(-1)

            # q(z|x)
            mu = mu.to(device)
            sigma = sigma.to(device)
            normal = torch.distributions.Normal(mu, sigma)
            logqzx =  normal.log_prob(z).sum(-1)

            # p(z)
            mu_ = torch.zeros(L).to(device)
            std_ = torch.ones(L).to(device)
            normal = torch.distributions.Normal(mu_, std_)
            logpz = normal.log_prob(z).sum(-1)

            logpx[i, :]  = -logpxz - logqzx + logpz

        logpx_max = logpx.max(0)[0]
        estimated_logpx = torch.log(torch.exp(logpx - logpx_max[0])).sum(0) + logpx_max[0] - torch.log(torch.FloatTensor([K])).to(device)

    return estimated_logpx

def eval_log_estimate(model, loader, K=200, L=100):
    model.to(device)
    sum_logpx = 0
    with torch.no_grad():
        for idx, x in enumerate(loader):
            x = x.to(device)
            logpx = importance_sampling(model, x)
            sum_logpx += logpx.sum()
    estimated_sum_logpx = sum_logpx / len(loader.dataset)
    return estimated_sum_logpx

def eval_ELBO(model, loader):
    model.to(device)
    running_loss = 0
    with torch.no_grad():
        for idx, x in enumerate(loader):
            x = x.to(device)
            recon_x, mu, log_sigma, sigma = model(x)
            loss = ELBO(x, recon_x, mu, sigma)
            running_loss += loss.item()
    return running_loss / len(loader.dataset)

if __name__ == "__main__":

    # Load dataset
    print("Loading datasets.....")
    start_time = time.time()
    train_set = MNIST("data", split="train")
    valid_set = MNIST("data", split="valid")
    test_set = MNIST("data", split="test")
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    print("DONE in {:.2f} sec".format(time.time() - start_time))

    do_training = False
    do_testing = True

    if do_training:
        # Set hyperparameters
        model = VAE(L=100)
        t_loss, v_loss = train(model, valid_loader, test_loader)
        torch.save(model.state_dict(), "vae.pth")

    if do_testing:
        print(device)
        model = VAE(L=100)
        model.load_state_dict(torch.load("vae.pth"))
        model.to(device)
        model.eval()
        print(len(valid_loader.dataset))

        valid_elbo = eval_ELBO(model, valid_loader)
        test_elbo = eval_ELBO(model, test_loader)
        print('Valid ELBO                    :   {:.4f}'.format(-valid_elbo))
        print('Test ELBO                     :   {:.4f}'.format(-test_elbo))

        valid_esl = eval_log_estimate(model, valid_loader)
        test_esl = eval_log_estimate(model, test_loader)
        print('Valid Estimated Log-likelihood:   {:.4f}'.format(-valid_esl))
        print('Test Estimated Log-likelihood :   {:.4f}'.format(-test_esl))
