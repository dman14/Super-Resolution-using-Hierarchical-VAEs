# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
def compute_conv_dim(dim_size, kernel_size, padding, stride):
    return int((dim_size - kernel_size + 2 * padding) / stride + 1)

def compute_maxpool_dim(dim_size):
    return int( ( (dim_size + 2*maxpool_padding - maxpool_dilation*(maxpool_kernel1-1) - 1 )/maxpool_stride) + 1)


class CNN_SR(nn.Module):
    def __init__(self, n_1 = 64, f_1 = 9, n_2 = 32, f_2 = 1, f_3 = 5,
                channels = 3, height = 224, width = 224):

        super(CNN_SR, self).__init__()

        # Typical and basic setting according to the paper:
        # f_1 = 9, f_2 = 1, f_3 = 5, n_1 = 64, n_2 = 32

        #channels = 3 #trainset.data.shape[3] #change accordingly to the input
        #height = 32 #trainset.data.shape[1] # change accordingly
        #width = 32 #trainset.data.shape[2] # change accordingly
        stride = 1 # [stride_height, stride_width]

        padding_1 = f_1 // 2 # to keep the same pixel size of the image (stride is 1 as well), // means floor
        padding_2 = f_2 // 2
        padding_3 = f_3 // 2
        
        # Patch extraction and representation layer
        self.conv_1   = Conv2d(in_channels=channels,
                               out_channels=n_1,
                               kernel_size=f_1,
                               stride=stride,
                               padding=padding_1)
        self.conv1_out_height = compute_conv_dim(height, f_1, padding_1, stride)
        self.conv1_out_width = compute_conv_dim(width, f_1, padding_1, stride)

        # Non-linear mapping layer 
        self.conv_2   = Conv2d(in_channels=n_1,
                               out_channels=n_2,
                               kernel_size=f_2,
                               stride=stride,
                               padding=padding_2)
        
        self.conv2_out_height = compute_conv_dim(self.conv1_out_height, f_2, padding_2, stride)
        self.conv2_out_width = compute_conv_dim(self.conv1_out_width, f_2, padding_2, stride)


        # Reconstruction layer
        self.conv_3   = Conv2d(in_channels=n_2,
                               out_channels=channels,
                               kernel_size=f_3,
                               stride=stride,
                               padding=padding_3)
        

    def forward(self, x):
        x = relu(self.conv_1(x))
        x = relu(self.conv_2(x))
        x = self.conv_3(x)
        return x 

# net = CNN_SR()
# if torch.cuda.is_available():
#     print('##converting network to cuda-enabled')
#     net.cuda()
# print(net)


#Test the forward pass with dummy data

def Dummy_image(batch = 5, channels = 3, height = 32, width = 32):
  x = np.random.normal(0,1, (batch, channels, height, width)).astype('float32')
  x = Variable(torch.from_numpy(x))
  return x

# x = Dummy_image()
# x = x.cuda()
# output = net(x)
# print([x.size() for x in output])

def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if torch.cuda.is_available():
        return x.cuda()
    return x

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if torch.cuda.is_available():
        return x.cpu().data.numpy()
    return x.data.numpy()
