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
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from git.vae_sr import *

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

def training_cnn(net, train_loader, test_loader, num_epochs = 100 ):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tb = SummaryWriter()
    #images, labels = next(iter(train_loader))
    #images = images.to(device)
    #labels = labels.to(device)
    #grid = make_grid(images)
    #tb.add_image("images", grid)
    #tb.add_graph(net, images)

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
            print('outputs = ', outputs.shape)

            # note, target is the original tensor, as we're working with auto-encoders
            loss = loss_function(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())

        train_loss.append(np.mean(batch_loss))


        tb.add_scalar("Training Loss", train_loss[-1], epoch)

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

            loss = loss_function(outputs, y)

            valid_loss.append(loss.item())
            tb.add_scalar("Validation Loss", valid_loss[-1], epoch)

        if epoch == 0:
            continue
            
        print("on epoch:",epoch)#,"training loss:"train_loss[-1],"and validation loss:",valid_loss[-1])
    tb.close()
    print(train_loss)
    print(valid_loss)

def vae_init(dataloader):

    # define the models, evaluator and optimizer

    images, low_res = next(iter(dataloader))

    # VAE
    latent_features = 24
    vae = VariationalAutoencoder(images[0].shape, latent_features)

    # Evaluator: Variational Inference
    beta = 1
    vi = VariationalInference(beta=beta)

    # The Adam optimizer works really well with VAEs.
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    # define dictionary to store the training curves
    training_data = defaultdict(list)
    validation_data = defaultdict(list)
    return vae, training_data, validation_data, vi, optimizer

        
def training_vae(train_loader, test_loader, num_epochs = 100 ):
    a=0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f">> Using device: {device}")

    vae, training_data, validation_data, vi, optimizer=vae_init(dataloader = train_loader)

    # move the model to the device
    vae = vae.to(device)


    tb = SummaryWriter()
    #images, labels = next(iter(train_loader))
    #images = images.to(device)
    #labels = labels.to(device)
    #grid = make_grid(images)
    #tb.add_image("images", grid)
    #tb.add_graph(vae, images)
    
    # training..
    epoch =0
    while epoch < num_epochs:
        epoch+= 1
        training_epoch_data = defaultdict(list)
        vae.train()
        
        # Go through each batch in the training dataset using the loader
        # Note that y is not necessarily known as it is here
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            
            # perform a forward pass through the model and compute the ELBO
            loss, diagnostics, outputs = vi(vae, x , y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # gather data for the current bach
            for k, v in diagnostics.items():
                training_epoch_data[k] += [v.mean().item()]
                

        # gather data for the full epoch
        for k, v in training_epoch_data.items():
            training_data[k] += [np.mean(training_epoch_data[k])]

        # Evaluate on a single batch, do not propagate gradients
        with torch.no_grad():
            vae.eval()
            
            # Just load a single batch from the test loader
            x, y = next(iter(test_loader))
            x = x.to(device)
            y = y.to(device)
            
            # perform a forward pass through the model and compute the ELBO
            loss, diagnostics, outputs = vi(vae, x, y)
            
            # gather data for the validation step
            for k, v in diagnostics.items():
                validation_data[k] += [v.mean().item()]
        
        # Reproduce the figure from the begining of the notebook, plot the training curves and show latent samples
        #make_vae_plots(vae, x, y, outputs, training_data, validation_data)
        if a==0:
            plt.ion()
            fig, axes = plt.subplots(1, 2, figsize=(13,5), squeeze=False)
            a=1
            ax = axes[0, 0]
            ax2 = axes[0, 1]

        tb.add_scalar("Training Loss", training_data['elbo'][-1], epoch)
        tb.add_scalar("Training Loss", training_data['kl'][-1], epoch)
        tb.add_scalar("Training Loss", validation_data['elbo'][-1], epoch)
        tb.add_scalar("Training Loss", validation_data['kl'][-1], epoch)

        print("epoch:",epoch)
    return vae
