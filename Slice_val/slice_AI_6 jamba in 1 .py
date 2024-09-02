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

from jamba.model import Jamba,JambaBlock

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

model_path = '../model/Mamba_best_model_300.pth'
output_path = '../output/AIData_ALL_model_jamba_300_6_1.csv'

feature_columns = ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' , 'Mass Flux', 'Inlet Subcooling']


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
# data_from_csv = pd.read_csv(slice_file_path)
# data_from_csv = pd.read_excel(slice_real_file_path, sheet_name='chf_public_LUT', engine='openpyxl')

allData = readDataIfExist(public_sheet_name)
numeric_data = allData.iloc[1:].to_numpy(dtype=float)

# Removing the last column which contains NaN values
numeric_data = numeric_data[:, :-1]

# Modifying the 'Pressure' column by dividing by 1e3
# numeric_data[:, 2] /= 1e3
# print(numeric_data)
slice_parameter = numeric_data[:, slice_parameter_index]
print(f'slice_parameter{slice_parameter}')

slice_data_2d = numeric_data


# Load the CSV file, skipping the first two header lines
# file_path = '../data/chf_public.csv'
# data = pd.read_csv(file_path, header=None)
allData1 = readDataIfExist(public_sheet_name)
# allData1.drop(allData1.columns[-1], axis=1, inplace=True)

# Set the column names using the first row and remove the first two rows
column_names = allData1.iloc[0].tolist()
# print(allData1.columns)
# print(column_names)
# allData1.columns = column_names
# print(allData1)
allData1 = allData1.drop(0)

# data_to_transform : 从第三列开始的所有数据列
data_to_transform = allData1.iloc[:, 2:]

# print(data_to_transform)
data_to_transform = data_to_transform.drop(['CHF','CHF LUT'], axis=1)
# print(data_to_transform)


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

# 将归一化后的数据组合成一个新的DataFrame
# normalized_data_to_transform = pd.DataFrame(normalized_columns)
# print(normalized_data_to_transform)
# print(scalers)

# 应用预先定义的scalers对slice_data_2d的每一列进行归一化
normalized_slice_data_2d =  data_to_transform[feature_columns].values



# 遍历每一列，应用相应的归一化容器
for i, column_name in enumerate(scalers):
    # print(column_name)
    normalized_slice_data_2d[:, i] = scalers[column_name].transform(normalized_slice_data_2d[:, i].reshape(-1, 1)).flatten()

print(normalized_slice_data_2d)

# slice_real_data = pd.read_excel(slice_real_file_path, sheet_name=sheet_name, engine='openpyxl')
slice_real_data = readDataIfExist(sheet_name)
slice_real_data['Enthalpy value of water'] = slice_real_data.apply(lambda x : calculateEnthalpy(x['Pressure'])[0],axis=1)
slice_real_data['Enthalpy value of air'] = slice_real_data.apply(lambda x : calculateEnthalpy(x['Pressure'])[1],axis=1)
slice_real_data['Enthalpy Temperature'] = slice_real_data.apply(lambda x : calculateEnthalpyTemperature(x['Pressure']),axis=1)


slice_real_data_values = slice_real_data[feature_columns].values


for i, column_name in enumerate(scalers):
    slice_real_data_values[:, i] = scalers[column_name].transform(slice_real_data_values[:, i].reshape(-1, 1)).flatten()


# ————————————————————————————————————————————————————————————加载模型————————————————————————————————————————————————————————————

# 定义Mamba模型
class JambaModel(nn.Module):
    def __init__(self, input_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(JambaModel, self).__init__()

        self.linear_in = nn.Linear(input_features, d_model)

        # dim=512,
        # depth=6 -> num_encoder_layers 2,
        # num_tokens=100,
        # d_state=256 ->64,
        # d_conv=128,
        # heads=8,
        # num_experts=8,
        # num_experts_per_token=2

        self.jamba_layer = nn.ModuleList(
            [
                JambaBlock(
                    d_model,
                    64,
                    128,
                    nhead,
                    8,
                    2,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.linear_f2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # 减小到 256 -> 1
        self.linear_mid2 = nn.Linear(d_model, d_model)
        self.linear_mid3 = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, output_size)


    def forward(self, src):
        # print(src.size())
        src = self.linear_in(src)
        # Apply the layers
        for layer in self.jamba_layer:
            src = layer(src)
        output = src
        # 通过额外的线性层和激活函数
        # output = F.gelu(self.linear_f1(output))
        output = F.softsign(self.linear_f2(output))
        output = self.dropout(output)
        output = F.softsign(self.linear_mid2(output))
        output = F.softsign(self.linear_mid3(output))
        output = self.linear_out(output)
        return output

# 超参数的搜索范围
input_size = len(feature_columns)
output_size = 1
batch_size = 256
d_model = 16
nhead = 8
num_encoder_layers = 5
dim_feedforward = 4096
dropout = 0.01
last_batch_size = batch_size # 最后一个批次大小
current_batch_size = batch_size
different_batch_size = False


model = JambaModel(input_features=input_size, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward, dropout=dropout)


# 加载保存的模型权重
pth = torch.load(model_path,map_location=device)
model.load_state_dict(pth)
model.eval()  # 设置为评估模式

# ————————————————————————————————————————————————————————————得到结果————————————————————————————————————————————————————————————

normalized_slice_data_2d = normalized_slice_data_2d.astype(float)
input_data = torch.tensor(normalized_slice_data_2d, dtype=torch.float)
input_data = input_data.unsqueeze(0)


input_data_2 = torch.tensor(slice_real_data_values, dtype=torch.float)
input_data_2 = input_data_2.unsqueeze(0)

print(input_data)

print(input_data_2)

with torch.no_grad():
    output = model(input_data)
output = output.numpy()


with torch.no_grad():
    output_2 = model(input_data_2)
output_2 = output_2.numpy()


# Extract the StandardScaler or MinMaxScaler
# file_path = '../data/chf_public.csv'
# data = pd.read_csv(file_path, header=None)
data = readDataIfExist(public_sheet_name)
data.drop(data.columns[-1], axis=1, inplace=True)
# Set the column names using the first row and remove the first two rows
# column_names = data.iloc[0].tolist()
# data.columns = column_names
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
output_2 = output_2.reshape(-1, 1)
# 使用最后一列的StandardScaler实例进行逆变换
output = scaler_standard[last_column_name].inverse_transform(output)
output_2 = scaler_standard[last_column_name].inverse_transform(output_2)

print(output_2)
print(output)

output = np.clip(output, 50, 16339)

# 转换为一维数组
out_put_float = output.reshape(-1)
outX = ['kW/m^2'] + out_put_float.tolist()
print(outX)
data = readDataIfExist(public_sheet_name)
data['AI CHF'] = outX
data.to_csv(output_path, index=False)


# ————————————————————————————————————————————————————————————画图————————————————————————————————————————————————————————————


# 使用不同的颜色和标记绘制所有 11 个二维矩阵在一个图上
# plt.figure(figsize=(12, 8))
# plt.plot(slice_parameter, output[:, 0], marker='o', color='red', label='AI Data', linewidth=2, markersize=8)

# 读取CSV文件

data_lut = readDataIfExist(public_sheet_name)
# 获取最后一列的数据，排除第一个值
last_column_data = data_lut.iloc[:, -1].values[1:].astype(float) / 1000

print(last_column_data)
# 添加LUT数据，假设横坐标与之前的Tube Diameter相同
plt.plot(slice_parameter, last_column_data, marker='x', color='black', label='LUT data', linewidth=2, markersize=8, zorder=1)

plt.scatter(slice_real_data[slice_parameter_name_1], slice_real_data['CHF'], color='purple', marker='D',  label='Real data', zorder=2)

# Setting title and labels with increased font size
plt.title(slice_parameter_name_1+' vs pre CHF', fontsize=16)
plt.xlabel(slice_parameter_name, fontsize=14)
plt.ylabel('Pre CHF [kW/m^2]', fontsize=14)

# Enlarging the tick marks' font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

output_indicator("LUT vs Real", slice_real_data['CHF LUT'], slice_real_data['CHF'])
output_indicator("AI vs Real", output_2.flatten(), slice_real_data['CHF'])