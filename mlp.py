import os
import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import pdb


forecast_length = 1
backcast_length = 10
batch_size = 4


# The class defining the multilayer perceptron architecture for training.
class MLP(nn.Module):
  def __init__(self, input_dim, output_dim):
      super().__init__()
              
      self.input_fc = nn.Linear(input_dim, 250)
      self.hidden_fc = nn.Linear(250, 100)
      self.output_fc = nn.Linear(100, output_dim)
      
  def forward(self, x):
      #x = [batch size, height, width]
      h_1 = F.relu(self.input_fc(x))
      #h_1 = [batch size, 250]
      h_2 = F.relu(self.hidden_fc(h_1))
      #h_2 = [batch size, 100]
      y_pred = self.output_fc(h_2)
      #y_pred = [batch size, output dim]
      return y_pred, h_2


# The data is being splitted into batches following the batch size variable defined at top of file.
def data_generator(x, y, size):
    assert len(x) == len(y)
    batches = []
    for ii in range(0, len(x), size):
        batches.append((x[ii:ii + size], y[ii:ii + size]))
    for batch in batches:
        yield batch


# 
def training(x_train, y_train, x_test, y_test):
# Calling the class initializer for compiling the model for training.
  model = MLP(backcast_length, forecast_length)
# Defining loss function for determining the loss after each epoch.
  loss_function = nn.CrossEntropyLoss()
# Defining the optimizer which tries to decrease the loss during training of the model.
  optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop over epochs.
  grad_step = 0
  for epoch in range(400):
    # train.
    model.train()
    train_loss = []
    for x_train_batch, y_train_batch in data_generator(x_train, y_train, batch_size):
      grad_step += 1
      optimiser.zero_grad()
      _, forecast = model(torch.tensor(x_train_batch, dtype=torch.float))
      loss = F.mse_loss(forecast, torch.tensor(y_train_batch, dtype=torch.float))
      train_loss.append(loss.item())
      loss.backward()
      optimiser.step()
      print(np.mean(train_loss))
