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

from git.helper import *


transform = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])


def init_dataloader(train_path, test_path, batch_size = 32, scale = 4):

  training_dataset = datasets.ImageFolder(train_path, transform=transform)
  test_dataset = datasets.ImageFolder(test_path, transform=transform)


  image_sets =[]
  for i in range(len(training_dataset)):
      image_sets.append(rescale(training_dataset[i][0],scale = scale))

  trainloader = torch.utils.data.DataLoader(image_sets, shuffle=True, batch_size= batch_size)

  image_sets2 =[]
  for i in range(len(test_dataset)):
      image_sets2.append(rescale(test_dataset[i][0],scale = scale))

  testloader = torch.utils.data.DataLoader(image_sets2, shuffle=True, batch_size= batch_size)
  

  return trainloader, testloader