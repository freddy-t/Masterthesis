# https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
from torchviz import make_dot

# Data Generation
np.random.seed(42)
x = np.random.rand(100, 1)
y = 1 + 2 * x + .1 * np.random.randn(100, 1)

# Shuffles the indices
idx = np.arange(100)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:80]
# Uses the remaining indices for validation
val_idx = idx[80:]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]


# ## regression with numpy
#
# # Initializes parameters "a" and "b" randomly
# np.random.seed(42)
# a = np.random.randn(1)
# b = np.random.randn(1)
#
# print(a, b)
#
# # Sets learning rate
# lr = 1e-1
# # Defines number of epochs
# n_epochs = 1000
#
# for epoch in range(n_epochs):
#     # Computes our model's predicted output
#     yhat = a + b * x_train
#
#     ## forward pass
#     # How wrong is our model? That's the error!
#     error = (y_train - yhat)
#     # It is a regression, so it computes mean squared error (MSE)
#     loss = (error ** 2).mean()
#
#     ## backward pass
#     # Computes gradients for both "a" and "b" parameters
#     a_grad = -2 * error.mean()
#     b_grad = -2 * (x_train * error).mean()
#
#     # Updates parameters using gradients and the learning rate
#     a = a - lr * a_grad
#     b = b - lr * b_grad
#
# print(a, b)

########################################################################################################################

# cuda just needs to be used for GPU usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
# and then we send them to the chosen device
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

# Here we can see the difference - notice that .type() is more useful
# since it also tells us WHERE the tensor is (device)
print(type(x_train), type(x_train_tensor), x_train_tensor.type())

########################################################################################################################
## final model


class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        # Computes the outputs / predictions
        return self.a + self.b * x


train_data = TensorDataset(x_train_tensor, y_train_tensor)
print('train data: {}'.format(train_data[0]))

train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(y, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    # Es wird eine Funktion zurückgegeben!!!
    return train_step


######################################################################################################
torch.manual_seed(42)

x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# Builds dataset with ALL data
dataset = TensorDataset(x_tensor, y_tensor)
# Splits randomly into train and validation datasets
train_dataset, val_dataset = random_split(dataset, [80, 20])
# Builds a loader for each dataset to perform mini-batch gradient descent
train_loader = DataLoader(dataset=train_dataset, batch_size=16)
val_loader = DataLoader(dataset=val_dataset, batch_size=20)

# Builds a simple sequential model
model = nn.Sequential(nn.Linear(1, 1)).to(device)
print(model.state_dict())

# Sets hyper-parameters
lr = 1e-1
n_epochs = 150

# Defines loss function and optimizer
loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)

losses = []
val_losses = []
# Creates function to perform train step from model, loss and optimizer
train_step = make_train_step(model, loss_fn, optimizer)

# Training loop
for epoch in range(n_epochs):
    # Uses loader to fetch one mini-batch for training
    for x_batch, y_batch in train_loader:
        # NOW, sends the mini-batch data to the device
        # so it matches location of the MODEL
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        # One stpe of training
        loss = train_step(x_batch, y_batch) ## train_step ist hier die Funktion, die direkt unter looses steht und nicht die oben definierte
        losses.append(loss)

    # After finishing training steps for all mini-batches,
    # it is time for evaluation!

    # We tell PyTorch to NOT use autograd...
    # bei der evaluation keine gradienten berechnen!
    with torch.no_grad():
        # Uses loader to fetch one mini-batch for validation
        for x_val, y_val in val_loader:
            # Again, sends data to same device as model
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            # model wird auf evaluation eingestellt, also ähnlich zu model.train()
            model.eval()
            # Makes predictions
            yhat = model(x_val)
            # Computes validation loss
            val_loss = loss_fn(y_val, yhat)
            val_losses.append(val_loss.item())

print(model.state_dict())
print(np.mean(losses))
print(np.mean(val_losses))
