import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.utils import data
from torchvision import datasets, transforms, utils
backends.cudnn.benchmark = True

import matplotlib.pyplot as plt
import numpy as np


# Code is inspired from: https://github.com/jzbontar/pixelcnn-pytorch/blob/master/main.py

device = ("cuda" if torch.cuda.is_available() else "cpu" )

print(f"device = {device}")

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

