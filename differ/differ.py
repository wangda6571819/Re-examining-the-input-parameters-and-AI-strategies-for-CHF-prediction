import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm



transfer_model_path = '../model/Transformer_best_model_101.pth'
new_model_path = '../model/Transformer_best_model_107.pth'

transfer_model = torch.load(transfer_model_path)
new_model = torch.load(new_model_path)


file_path = '../data/chf_public.csv'
data = pd.read_csv(file_path, header=None)
data.drop(data.columns[-1], axis=1, inplace=True)
# Set the column names using the first row and remove the first two rows
column_names = data.iloc[0].tolist()
data.columns = column_names
data = data.drop([0, 1])

print(data)

input_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality']
output_column = 'CHF'

def predict(mode, input_data) :
	output = mode(input_data)
	prediction = torch.argmax(output).item()
	print(f'predict is {prediction}')


 # (D  L=[5.8, 6.05], G=[978, 1019], P=[14700, 14710], X=[0.378, 0.404], T=[36, 331])
# Tube Diameter	Heated Length	Pressure	Mass Flux	Outlet Quality (Inlet Subcooling)	Inlet Temperature
condition = {
	'D' : 'Tube Diameter',
	'L' : 'Heated Length',
	'P' : 'Pressure',
	'X' : 'Outlet Quality',
	'T' : 'Inlet Temperature',
	'H' : 'Inlet Subcooling',
	'G' : 'Mass Flux',
}

# (L=[5.8, 6.05], G=[978, 1019], P=[14700, 14710], X=[0.378, 0.404], T=[36, 331])
queryString = f"{condition['D']} >= 5.8 and {condition['D']} <= 6.05 and {condition['G']} >= 978 and {condition['G']} <= 1019 and {condition['P']} >= 14700 and {condition['P']} <= 14710 and {condition['D']} >= 0.378 and {condition['D']} <=0.404 and {condition['T']} >= 36 and {condition['T']} <= 331"
dataQ = data.query(queryString)
print(dataQ)
