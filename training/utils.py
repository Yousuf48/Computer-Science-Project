import os
import time
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.utils import data
from torchvision import datasets, transforms, utils
backends.cudnn.benchmark = True


import matplotlib.pyplot as plt
import numpy as np



class AddGaussianNoise:
  def __init__(self, mean= 0, std=0.1):
    self.mean = mean
    self.std = std

  def __call__(self, x):
    x = x + torch.normal(self.mean, self.std, x.size())
    return torch.clamp(x, 0, 1)

  def __repr__(self):
    return f"mean= {self.mean} , std = {self.std}"


class PairDataset(data.Dataset):
    def __init__(self, clean_dataset, noisy_dataset):
        self.clean_dataset = clean_dataset
        self.noisy_dataset = noisy_dataset

    def __getitem__(self, index):
        # image, lable = dataset[index]
        clean_x, clean_y = self.clean_dataset[index]
        noisy_x, noisy_y = self.noisy_dataset[index]

        return clean_x, clean_y, noisy_x, noisy_y

    def __len__(self):
       return min(len(self.clean_dataset),len(self.noisy_dataset))




noisy_transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise()
    ])


noisy_transform_more = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0,0.3)
    ])