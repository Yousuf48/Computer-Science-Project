import os
import time
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms, utils
backends.cudnn.benchmark = True

from masked_conv import *
from utils import *


# A class combines both PixelCNN and NoisyCNN
class CombinedCNN(nn.Module):
    def __init__(self, pixel_cnn_channels, noisy_channels,combined_channels,*args, **kwargs):
      super(CombinedCNN, self).__init__(*args, **kwargs)

      self.pixel_cnn = nn.Sequential(
          MaskedConv2d('A', 1,  pixel_cnn_channels, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels), nn.ReLU(True),
          MaskedConv2d('B', pixel_cnn_channels, pixel_cnn_channels, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels), nn.ReLU(True),
          MaskedConv2d('B', pixel_cnn_channels, pixel_cnn_channels, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels), nn.ReLU(True),
          MaskedConv2d('B', pixel_cnn_channels, pixel_cnn_channels, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels), nn.ReLU(True),
          MaskedConv2d('B', pixel_cnn_channels, pixel_cnn_channels *2, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels*2), nn.ReLU(True),
          MaskedConv2d('B', pixel_cnn_channels*2, pixel_cnn_channels *2, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels*2), nn.ReLU(True),
          MaskedConv2d('B', pixel_cnn_channels*2, pixel_cnn_channels *2, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels*2), nn.ReLU(True),
          MaskedConv2d('B', pixel_cnn_channels*2, pixel_cnn_channels *2, 7, 1, 3, bias=False), nn.BatchNorm2d(pixel_cnn_channels*2), nn.ReLU(True),
          nn.Conv2d(pixel_cnn_channels*2, 256, 1)
      )

      self.noisy_cnn = nn.Sequential(
          nn.Conv2d(1, noisy_channels, 3, padding=1),
          nn.ReLU(True),
          nn.Conv2d(noisy_channels, noisy_channels*2, 3,padding=1),
          nn.ReLU(True),
          nn.Conv2d(noisy_channels*2, noisy_channels*4, 3, padding=1),
          nn.ReLU(True),
          nn.Conv2d(noisy_channels*4, noisy_channels*4, 3, padding=1),
          nn.ReLU(True),
          nn.Conv2d(noisy_channels*4, 256, 1)
          )

      self.combined_cnn = nn.Sequential(
          nn.Conv2d(512,combined_channels, 1),
          nn.ReLU(True),

          nn.Conv2d(combined_channels, combined_channels, 1),
          nn.ReLU(True),

          nn.Conv2d(combined_channels, combined_channels*2, 1),
          nn.ReLU(True),

          nn.Conv2d(combined_channels*2, combined_channels*2, 1),
          nn.ReLU(True),

          nn.Conv2d(combined_channels*2, combined_channels*2, 1),
          nn.ReLU(True),

          nn.Conv2d(combined_channels*2, 256, 1)
          )

    def forward(self, x, y):
        x = self.pixel_cnn(x)
        y = self.noisy_cnn(y)
        c = torch.cat((x,y), dim=1)

        c = self.combined_cnn(c)

        return c

combined_cnn = CombinedCNN(64,32,64).to(device)
print(combined_cnn)




clean_training_datasets_fashion  = datasets.FashionMNIST('datasets',
                   train=True,
                   download=True,
                   transform= transforms.ToTensor()
                   )


noisy_training_datasets_fashion  = datasets.FashionMNIST('datasets',
                   train=True,
                   download=True,
                   transform= noisy_transform_more
                   )




pair_datasets_fashion = PairDataset(clean_training_datasets_fashion, noisy_training_datasets_fashion )


optimizer = optim.Adam(combined_cnn.parameters(), lr=0.001)

images_pair = data.DataLoader(pair_datasets_fashion, batch_size=128,num_workers=4 , pin_memory=True,shuffle=True)

epochs = 40



losses = []
for i in range(epochs):
  print(i)
  los = []
  time_ = time.time()
  combined_cnn.train()
  for img_c, _, img_n, _ in images_pair:
    img_c, img_n = img_c.to(device), img_n.to(device)
    inputs = combined_cnn(img_c, img_n)
    
    target = (img_c[:,0] * 255).long()
    loss = F.cross_entropy(inputs, target)
    
    losses.append(loss.detach().item())
    los.append(loss.detach().item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f"loss: {np.mean(los)}, time:{time.time() -  time_} ") 

torch.save(combined_cnn.state_dict(), "models/denosing_more_level.pth")
np.save("denoising_level.npy",losses )



