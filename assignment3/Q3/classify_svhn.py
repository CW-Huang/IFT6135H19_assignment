#!/usr/bin/env python
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import dataset
from torch import nn
# from torch.nn.modules import upsampling
# from torch.functional import F
from torch.optim import Adam

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5),
                         (.5, .5, .5))
])


def get_data_loader(dataset_location, batch_size):
    trainvalid = torchvision.datasets.SVHN(
        dataset_location, split='train',
        download=True,
        transform=image_transform
    )

    trainset_size = int(len(trainvalid) * 0.9)
    trainset, validset = dataset.random_split(
        trainvalid,
        [trainset_size, len(trainvalid) - trainset_size]
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

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            dataset_location, split='test',
            download=True,
            transform=image_transform
        ),
        batch_size=batch_size,
    )

    return trainloader, validloader, testloader


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 512, 2),
        )

        self.mlp = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, x):
        return self.mlp(self.extract_features(x))

    def extract_features(self, x):
        return self.conv_stack(x)[:, :, 0, 0]


def evaluate(classify, dataset):
    with torch.no_grad():
        classify.eval()
        correct = 0.
        total = 0.
        for x, y in dataset:
            if cuda:
                x = x.cuda()
                y = y.cuda()

            c = (classify(x).argmax(dim=-1) == y).sum().item()
            t = x.size(0)
            correct += c
            total += t
    acc = correct / float(total)
    return acc


if __name__ == "__main__":
    train, valid, test = get_data_loader("svhn", 32)
    classify = Classifier()
    params = classify.parameters()
    optimizer = Adam(params)
    ce = nn.CrossEntropyLoss()
    best_acc = 0.
    cuda = torch.cuda.is_available()
    if cuda:
        classify = classify.cuda()

    for _ in range(50):
        classify.train()
        for i, (x, y) in enumerate(train):
            if cuda:
                x = x.cuda()
                y = y.cuda()
            out = classify(x)
            loss = ce(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i + 1) % 200 == 0:
                print(loss.item())
        acc = evaluate(classify, valid)
        print("Validation acc:", acc,)

        if acc > best_acc:
            best_acc = acc
            torch.save(classify, "svhn_classifier.pt")
            print("Saved.")
    classify = torch.load("svhn_classifier.pt")
    print("Test accuracy:", evaluate(classify, test))
