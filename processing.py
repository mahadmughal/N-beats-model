import pandas as pd
import numpy as np



def load_file(file_path):
  dataframe = pd.read_csv(file_path)

  return dataframe





dataframe = load_file('tourism_dataset/monthly_in.csv')
