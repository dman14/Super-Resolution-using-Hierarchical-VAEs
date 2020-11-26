import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

training_init():

    # if you want L2 regularization, then add weight_decay to SGD
    optimizer = optim.SGD(net.parameters(), lr=0.025)

    # We will use pixel wise mean-squared error as our loss function
    loss_function = nn.MSELoss()

    return optimizer, loss_function

def test_network(net, trainloader):

    optimizer, loss_function = training_init()


    dataiter = iter(trainloader)
    images, labels = dataiter.next()

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

def training(net, num_epochs = 100, ):

    train_loss = []
    valid_loss = []

    optimizer, loss_function = training_init()
    

    for epoch in range(num_epochs, train_loader):
        batch_loss = []
        net.train()
        
        # Go through each batch in the training dataset using the loader
        # Note that y is not necessarily known as it is here
        for x, y in train_loader:
            
            if cuda:
                x = x.cuda()
            
            outputs = net(x)
            x_hat = outputs['x_hat']

            # note, target is the original tensor, as we're working with auto-encoders
            loss = loss_function(x_hat, x)
            
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
            
            if cuda:
                x = x.cuda()
            
            outputs = net(x)

            # We save the latent variable and reconstruction for later use
            # we will need them on the CPU to plot
            x_hat = outputs['x_hat']
            z = outputs['z'].cpu().numpy()

            loss = loss_function(x_hat, x)

            valid_loss.append(loss.item())
        
        if epoch == 0:
            continue

        # live plotting of the trainig curves and representation
        plot_autoencoder_stats(x=x.cpu(),
                            x_hat=x_hat.cpu(),
                            z=z,
                            y=y,
                            train_loss=train_loss,
                            valid_loss=valid_loss,
                            epoch=epoch,
                            classes=classes,
                            dimensionality_reduction_op = None) #lambda z: TSNE(n_components=2).fit_transform(z))
        
    
