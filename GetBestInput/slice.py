
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from joblib import Parallel, delayed
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
from iapws import IAPWS97
import os

import argparse


model_path = '../model/Transformer_best_model_119.pth'
output_path = '../output/AIData_ALL_model_transformer_119_6_1(1000).csv'

feature_columns = ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' , 'Mass Flux', 'Inlet Subcooling']
feature_columns_string = ','.join(feature_columns)

# 创建解析器对象
parser = argparse.ArgumentParser(description='slice')

# 获取执行参数
# 添加一个可选参数
parser.add_argument("--featureColumns",default=feature_columns_string)
parser.add_argument("--modelPath",default=model_path)
parser.add_argument("--outputPath",default=output_path)
args = parser.parse_args()
print(args)

feature_columns = args.featureColumns.split(',')
model_path = args.modelPath
output_path = args.outputPath



pd.set_option('display.width', 50000) # 设置字符显示宽度
pd.set_option('display.max_rows', None) # 设置显示最大行
pd.set_option('display.max_columns', None) # 设置显示最大列，None为显示所有列


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def output_indicator(phase, pre, real):
    print("-------------------------------"+str(phase)+"-------------------------------")

    print("Mean P/M:", np.average(pre/ real))
    print("Std P/M:", np.std(pre / real))
    print("RMSPE:",  np.sqrt(np.mean(np.square((pre - real) / real))))
    print("MAPE:",  np.mean(np.abs((pre - real) / real)))
    # NRMSE - Normalized by the range of actual values
    rmse = np.sqrt(np.mean(np.square(real - pre)))
    nrmse_mean = rmse / np.mean(real)
    print("NRMSE:", nrmse_mean)
    # 计算 mu，即 Y 的平均值
    mu = np.mean(real)
    # 计算分子和分母
    numerator = np.sum((real - pre) ** 2)
    denominator = np.sum((real - mu) ** 2)
    # 计算 EQ^2
    EQ2 = numerator / denominator
    print("Q2:",  EQ2)

slice_real_file_path = '../data/slice_real_value (copy).xlsx'

data_file_path = '../data/slice_real_value (copy).xlsx'
public_sheet_name = 'chf_public_LUT'
sheet_name = 'slice02'
input_path = '../input/'




def readDataIfExist(name) :
    pkl_name = input_path + name + '.pkl'

    exist = os.path.exists(pkl_name)
    if exist :
        data  = pd.read_pickle(pkl_name)
        return data

    # 读取xlsx 获取数据
    data = pd.read_excel(data_file_path, sheet_name=name, engine='openpyxl')
    data.to_pickle(pkl_name) #格式另存
    return data

slice_parameter_name = 'Tube Diameter [m]'
slice_parameter_name_1 = 'Tube Diameter'
slice_parameter_index = 0
input_features = len(feature_columns)
# ————————————————————————————————————————————————————————————数据处理————————————————————————————————————————————————————————————
allData = readDataIfExist(public_sheet_name)
numeric_data = allData.iloc[1:].to_numpy(dtype=float)

numeric_data = numeric_data[:, :-1]

slice_parameter = numeric_data[:, slice_parameter_index]
print(f'slice_parameter{slice_parameter}')
slice_data_2d = numeric_data


# Load the CSV file, skipping the first two header lines
allData1 = readDataIfExist(public_sheet_name)

# Set the column names using the first row and remove the first two rows
column_names = allData1.iloc[0].tolist()
allData1 = allData1.drop(0)

# data_to_transform : 从第三列开始的所有数据列
data_to_transform = allData1.iloc[:, 2:]
data_to_transform = data_to_transform.drop(['CHF','CHF LUT'], axis=1)


# Enthalpy value of water
# Enthalpy value of air
def calculateEnthalpy(pressure):
    """
    根据压力计算水焓或气焓（kind='L'或'V'）
    :param kind: 字符串，'L'表示水焓，'V'表示气焓
    :param pressure: 压力值（kPa）
    :return: 水焓或气焓（kJ/kg）
    """
    pressure = float(pressure)
    vapor = IAPWS97(P= pressure/1000, x= 1.0)
    HVapor = vapor.h

    vaporH = IAPWS97(P= pressure/1000, x= 0)
    hLiquid = vaporH.h
    return [hLiquid,HVapor]

def calculateEnthalpyTemperature(pressure) :
    pressure = float(pressure)
    vapor = IAPWS97(P= pressure/1000, x= 1.0)
    return vapor.T

data_to_transform['Enthalpy value of water'] = data_to_transform.apply(lambda x : calculateEnthalpy(x['Pressure'])[0],axis=1)
data_to_transform['Enthalpy value of air'] = data_to_transform.apply(lambda x : calculateEnthalpy(x['Pressure'])[1],axis=1)
data_to_transform['Enthalpy Temperature'] = data_to_transform.apply(lambda x : calculateEnthalpyTemperature(x['Pressure']),axis=1)


data_to_transform = data_to_transform.reindex(columns=feature_columns)

# 初始化StandardScaler的字典来存储每列的归一化容器
scalers = {}
# 对每列进行归一化
normalized_columns = {}
for column in data_to_transform.columns:
    print(column)
    scaler = StandardScaler()
    normalized_columns[column] = scaler.fit_transform(data_to_transform[[column]].values).flatten()
    scalers[column] = scaler

# 应用预先定义的scalers对slice_data_2d的每一列进行归一化
normalized_slice_data_2d =  data_to_transform[feature_columns].values

# 遍历每一列，应用相应的归一化容器
for i, column_name in enumerate(scalers):
    normalized_slice_data_2d[:, i] = scalers[column_name].transform(normalized_slice_data_2d[:, i].reshape(-1, 1)).flatten()

print(normalized_slice_data_2d)


# ————————————————————————————————————————————————————————————加载模型————————————————————————————————————————————————————————————


# Defining the neural network architecture
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.01, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.pe, mean=0, std=0.02)

    def forward(self, x):
        position = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        x = x + self.pe[position]
        return x
class TransformerModel(nn.Module):
    def __init__(self, input_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.linear_in = nn.Linear(input_features, d_model)
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_encoder = LearnablePositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # 更复杂的输出层
        self.linear_mid = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(d_model, 1)

        # 更复杂的输出层
        self.linear_mid1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear_mid2 = nn.Linear(d_model, d_model)
        self.linear_mid3 = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.linear_in(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # 通过额外的线性层和激活函数
        output = F.relu(self.linear_mid1(output))
        output = self.dropout(output)
        output = F.relu(self.linear_mid2(output))
        output = F.relu(self.linear_mid3(output))
        output = self.linear_out(output)
        return output


d_model = 64
nhead = 32
num_encoder_layers = 6
dim_feedforward = 4096
dropout = 0.01

model = TransformerModel(input_features=input_features, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward, dropout=dropout)


# 加载保存的模型权重
pth = torch.load(model_path,map_location=device)
model.load_state_dict(pth)
model.eval()  # 设置为评估模式

# ————————————————————————————————————————————————————————————得到结果————————————————————————————————————————————————————————————

normalized_slice_data_2d = normalized_slice_data_2d.astype(float)
input_data = torch.tensor(normalized_slice_data_2d, dtype=torch.float)
input_data = input_data.unsqueeze(0)

print('开始分段预测结果')
list_input_data = torch.chunk(input_data, dim = 1 , chunks = 3)

output = []
for idx, iData in enumerate(list_input_data) :

    with torch.no_grad():
        x = model(iData)
        x = x.numpy()
        if output == [] :
            output = x
        else :
            output = np.append(output, x , axis=1)

data = readDataIfExist(public_sheet_name)
data.drop(data.columns[-1], axis=1, inplace=True)
data = data.drop(0)

data = data.drop(['Pressure'], axis=1)
# print(data)
# data_to_transform : 从第三列开始的所有数据列
data_to_transform = data.iloc[:, 2:]

# 假设scaler_minmax是一个字典，保存了除第一列和第二列外的每列数据的MinMaxScaler实例
scaler_standard = {column: StandardScaler() for column in data_to_transform.columns}

# 假设在适当的地方以列为单位进行了fit_transform操作，比如
for column in scaler_standard:
    # 只针对不含'-'的列进行操作
    if column[1] != '-':
        scaler_standard[column].fit(data_to_transform[[column]])

# 获取最后一列的标签名称
last_column_name = data.columns[-1]

# 为了逆变换，需要确保它是二维的
output = output.reshape(-1, 1)
# 使用最后一列的StandardScaler实例进行逆变换
output = scaler_standard[last_column_name].inverse_transform(output)
print(output)

output = np.clip(output, 50, 16339)

# 转换为一维数组
out_put_float = output.reshape(-1)
outX = ['kW/m^2'] + out_put_float.tolist()
print(outX)
data = readDataIfExist(public_sheet_name)
data['AI CHF'] = outX
data.to_csv(output_path, index=False)
