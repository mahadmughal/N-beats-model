import pandas as pd
import numpy as np
import glob
import os
import pdb
from sklearn.preprocessing import MinMaxScaler
from nbeats_pytorch.model import NBeatsNet
import torch
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import n_beats


file_path = 'tourism_dataset/monthly_oos.csv'
forecast_length = 1
backcast_length = 10
split_ratio = 0.8


def data_preprocessing(file_path):
  dataframe = pd.read_csv(file_path)
  data = data_normalization(dataframe)
  data = data_interpolation(data)
  x, y = data_sequencing(data)
  x_train, y_train, x_test, y_test = data_splitting(x, y)

  return x_train, y_train, x_test, y_test


def data_normalization(arr):
  scaler = MinMaxScaler()
  scaler.fit(arr)
  arr = scaler.transform(arr)

  return arr.flatten()


def data_sequencing(data):
  x, y = [], []
  steps = 1
  for epoch in range(backcast_length, len(data)-forecast_length, steps):
    x.append(data[epoch - backcast_length:epoch])
    y.append(data[epoch:epoch + forecast_length])

  return x, y


def data_interpolation(data):
  data = data[~(np.isnan(data))]

  return data


def data_splitting(x, y):
  c = int(len(x) * 0.8)
  x_train, y_train = x[:c], y[:c]
  x_test, y_test = x[c:], y[c:]

  return x_train, y_train, x_test, y_test




x_train, y_train, x_test, y_test = data_preprocessing(file_path)
model, optimiser = n_beats.architecture(backcast_length, forecast_length)
n_beats.training(model, optimiser, x_train, y_train, x_test, y_test)
