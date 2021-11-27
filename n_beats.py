from nbeats_pytorch.model import NBeatsNet
import torch
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import pdb
from sklearn.metrics import r2_score


CHECKPOINT_NAME = 'n_beats_trained_model.th'

# forecase_length defined the count of outputs to predict given the input sample.
forecast_length = 2
# backcast_length defined the count of input features to predict the output.
backcast_length = 10
# Divides the dataset into batches for training.
batch_size = 4

# splitting the dataset into batches following the batch size variable defined early.

def data_generator(x, y, size):
    assert len(x) == len(y)
    batches = []
    for ii in range(0, len(x), size):
        batches.append((x[ii:ii + size], y[ii:ii + size]))
    for batch in batches:
        yield batch


# Save the model into the folder.
def save(model, optimiser, grad_step=0):
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, CHECKPOINT_NAME)


# Draw the scattering plots for visualisation.
def plot_scatter(*args, **kwargs):
    plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs)


# Defining the architecture of the N-beats model to train.


def architecture(backcast_length, forecast_length):
  model = NBeatsNet(
        stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
        forecast_length=forecast_length,
        backcast_length=backcast_length,
        hidden_layer_units=128,
    )
  optimiser = optim.Adam(lr=1e-4, params=model.parameters())

  return model, optimiser


# Training loop over number of epochs.
def training(model, optimiser, x_train, y_train, x_test, y_test):
  grad_step = 0
  for epoch in range(400):
    # train.
    model.train()
    train_loss = []
    for x_train_batch, y_train_batch in data_generator(x_train, y_train, batch_size):
      grad_step += 1
      optimiser.zero_grad()
      _, forecast = model(torch.tensor(x_train_batch, dtype=torch.float).to(model.device))
      loss = F.mse_loss(forecast, torch.tensor(y_train_batch, dtype=torch.float).to(model.device))
      train_loss.append(loss.item())
      loss.backward()
      optimiser.step()
    train_loss = np.mean(train_loss)

    model.eval()
    _, forecast = model(torch.tensor(x_test, dtype=torch.float))
    test_loss = F.mse_loss(forecast, torch.tensor(y_test, dtype=torch.float)).item()
    p = forecast.detach().numpy()
    test_score = r2_score(y_test, p)

    if epoch % 100 == 0:
      subplots = [221, 222, 223, 224]
      plt.figure(1)
      for plot_id, i in enumerate(np.random.choice(range(len(x_test)), size=4, replace=False)):
        ff, xx, yy = p[i] * 1, x_test[i] * 1, y_test[i] * 1
        plt.subplot(subplots[plot_id])
        plt.grid()
        plot_scatter(range(0, backcast_length), xx, color='b')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
      plt.savefig('plots/'+str(epoch)+'_image.png')

    with torch.no_grad():
        save(model, optimiser, grad_step)
    print(f'epoch = {str(epoch).zfill(4)}, '
          f'grad_step = {str(grad_step).zfill(6)}, '
          f'train_loss (epoch) = {1000 * train_loss:.3f}, '
          f'test_loss (epoch) = {1000 * test_loss:.3f}, ')


