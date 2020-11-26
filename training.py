from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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

def training_init(net):

    # if you want L2 regularization, then add weight_decay to SGD
    optimizer = optim.SGD(net.parameters(), lr=0.025)

    # We will use pixel wise mean-squared error as our loss function
    loss_function = nn.MSELoss()

    return optimizer, loss_function

def test_network(net, trainloader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = net.to(device)

    optimizer, loss_function = training_init(net)


    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    images = images.to(device)

    # Create Variables for the inputs and targets
    inputs = Variable(images)
    targets = Variable(images)

    # Clear the gradients from all Variables
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = net.forward(inputs)
    loss = loss_function(output, targets)
    loss.backward()
    optimizer.step()

    return True

def training(net, train_loader, test_loader, num_epochs = 100 ):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f">>using device: {device}")


    train_loss = []
    valid_loss = []

    optimizer, loss_function = training_init(net)
    net = net.to(device)
    

    for epoch in range(num_epochs):
        batch_loss = []
        net.train()
        
        # Go through each batch in the training dataset using the loader
        # Note that y is not necessarily known as it is here
        for x, y in train_loader:
            
            x = x.to(device)
            y = y.to(device)
            
            outputs = net(x)

            # note, target is the original tensor, as we're working with auto-encoders
            loss = loss_function(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())

        train_loss.append(np.mean(batch_loss))

        # Evaluate, do not propagate gradients
        with torch.no_grad():
            net.eval()
            
            # Just load a single batch from the test loader
            x, y = next(iter(test_loader))
            
            x = x.to(device)
            y = y.to(device)
            
            outputs = net(x)

            # We save the latent variable and reconstruction for later use
            # we will need them on the CPU to plot
            x_hat = outputs
            #z = outputs['z'].cpu().numpy()

            loss = loss_function(x_hat, x)

            valid_loss.append(loss.item())

        if epoch == 0:
            continue
        # live plotting of the trainig curves and representation
        #plot_autoencoder_stats(x=x.cpu(),
        #                    x_hat=x_hat.cpu(),
        #                    z=z,
        #                    y=y,
        #                    train_loss=train_loss,
        #                    valid_loss=valid_loss,
        #                    epoch=epoch,
        #                    classes=classes,
        #                    dimensionality_reduction_op = None) #lambda z: TSNE(n_components=2).fit_transform(z))
        print("on epoch:",epoch)

    print(train_loss)
    print(valid_loss)
    return net
        
    