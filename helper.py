# -*- coding: utf-8 -*-
# define helper.py 
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import PIL.Image as pil_image
    
def test_network(net, trainloader):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Create Variables for the inputs and targets
    inputs = Variable(images)
    targets = Variable(images)

    # Clear the gradients from all Variables
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = net.forward(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    return True


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

def rescale(image, scale=4):
  to_pil_image = transforms.ToPILImage()
  
  hr = to_pil_image(image)
  hr_width = (hr.width // scale) * scale
  hr_height = (hr.height // scale) * scale
  hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
  lr = hr.resize((hr_width // scale, hr_height // scale), resample=pil_image.BICUBIC)
  lr = lr.resize((lr.width * scale, lr.height * scale), resample=pil_image.BICUBIC)

  pil_to_tensor = transforms.ToTensor()(hr).unsqueeze_(0)
  tensor_to_pil = transforms.ToPILImage()(pil_to_tensor.squeeze_(0))
  hr = pil_to_tensor

  pil_to_tensor2 = transforms.ToTensor()(lr).unsqueeze_(0)
  tensor_to_pil2 = transforms.ToPILImage()(pil_to_tensor2.squeeze_(0))
  lr = pil_to_tensor2
  return(hr,lr)

def batchRescale(images=1, scale= 4):
    lr_batch = []
    for image in images:
        lr_batch.append(rescale(image,scale=4))
    
    return lr_batch


