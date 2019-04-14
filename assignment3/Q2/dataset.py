from torchvision.datasets import utils
import torch.utils.data as data_utils
import torch
import os
import numpy as np
from torch import nn
from torch.nn.modules import upsampling
from torch.functional import F
from torch.optim import Adam
from torch.utils.data import Dataset

def lines_to_np_array(lines):
    return np.array([[int(i) for i in line.split()] for line in lines])

class MNIST(Dataset):
    """
    Args:
        data_dir: Path to the directory containing the samples.
        split: Which split to use. [train, valid, test]
        subset: How many elements will be used. Default: all.
        skip: How many element to skip before taking the subset.
    """
    def __init__(self, data_dir, split="train", subset=None, skip=0):
        filename = "binarized_mnist_%s.amat" % split
        filepath = os.path.join(data_dir, filename)
        with open(filepath) as f:
            lines = f.readlines()
        x = lines_to_np_array(lines).astype('float32')
        x = x.reshape(x.shape[0], 1, 28, 28)
        self.data = torch.from_numpy(x)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return x
