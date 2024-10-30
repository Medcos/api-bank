import pandas as pd
import numpy as np
import matplotlib
import os
import sys

## Importer les données
data_path = os.path.join(os.getcwd(), 'info_clients.csv')
info_client = pd.read_csv(data_path)

df_path = os.path.join(os.getcwd(), 'data.csv')
data = pd.read_csv(df_path)

data_path = os.path.join(os.getcwd(), 'data_test.csv')
data_test = pd.read_csv(data_path)

print(data)