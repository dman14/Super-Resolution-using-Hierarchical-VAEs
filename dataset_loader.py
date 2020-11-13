import os
import numpy as np
import torch
import math
from torch import nn, Tensor
from torch.nn.functional import softplus
from torch.distributions import Distribution

from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


transform = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])


def init_dataloader(train_path, test_path, batch_size = 32):

  training_dataset = datasets.ImageFolder(train_path, transform=transform)
  test_dataset = datasets.ImageFolder(test_path, transform=transform)


  dataloader = torch.utils.data.DataLoader(training_dataset,
                                          batch_size=batch_size,
                                           shuffle=True)
  dataloader_test = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

  return dataloader, dataloader_test