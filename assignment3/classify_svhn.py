#!/usr/bin/env python
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import dataset
import os
import numpy as np
from torch import nn
from torch.nn.modules import upsampling
from torch.functional import F
from torch.optim import Adam


def get_data_loader(dataset_location, batch_size):
    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    trainset_size = int(len(trainvalid) * 0.8)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
    )

    testset = torchvision.datasets.SVHN(
        dataset_location, split='test',
        transform=None, target_transform=None,
        download=True
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
    )

    return trainloader, validloader



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 2),
        )

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.mlp(self.extract_features(x))

    def extract_features(self, x):
        return self.conv_stack(x).view(-1, 256)


if __name__ == "__main__":
    train, valid = get_data_loader("svhn", 32)
    classify = Classifier()
    params = classify.parameters()
    optimizer = Adam(params)
    ce = nn.CrossEntropyLoss()
    best_acc = 0.
    for _ in range(10):
        classify.train()
        for i, (x, y) in enumerate(train):
            out = classify(x)
            loss = ce(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % 200 == 0:
                print(loss.item())
        with torch.no_grad():
            classify.eval()
            correct = 0.
            total = 0.
            for x, y in valid:
                c = (classify(x).argmax(dim=-1) == y).sum().item()
                t = x.size(0)
                correct += c
                total += t
            acc = correct / float(total)
            print("Validation acc:", acc,)
            if acc > best_acc:
                torch.save(classify, "svhn_classifier.pt")
                print("Saved.")
            else:
                print()
