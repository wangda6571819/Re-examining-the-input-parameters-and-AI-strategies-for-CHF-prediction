
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

# data_file_path = '../output/AIData_ALL_model_116_9_1.csv'
# data_file_path = '../output/AIData_ALL_model_116_9_1.csv'
# data_file_path = '../output/AIData_ALL_model_116_9_1.csv'
# data_file_path = '../output/AIData_ALL_model_116_9_1.csv'
# data_file_path = '../output/AIData_ALL_model_116_9_1.csv'

model_name = 'transformer'
data_file_path = f'../output/AIData_ALL_model_{model_name}_119_6_1(1000).csv'
ori_file_path = '../data/slice_real_value (copy).xlsx'
# public_sheet_name = 'chf_public_LUT'
sheet_names = ['slice01','slice02','slice03','slice04','slice05','slice06','slice07','slice08','slice09','slice10']
input_path = '../input/'



data = pd.read_csv(data_file_path)
data.drop(data.index[0], inplace=True)
data = data.astype(float)


def output_indicator(phase, pre, real):
    print("-------------------------------"+str(phase)+"-------------------------------")

    mean = np.average(pre/ real)
    std = np.std(pre / real)
    rmspe = np.sqrt(np.mean(np.square((pre - real) / real)))
    mape = np.mean(np.abs((pre - real) / real))
    print("Mean P/M:", mean)
    print("Std P/M:", std)
    print("RMSPE:",  rmspe)
    print("MAPE:",  mape)
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
    return [ std , rmspe, mape , nrmse_mean, EQ2]

# condition = {
# 	'D' : '`Tube Diameter`',
# 	'L' : '`Heated Length`',
# 	'P' : '`Pressure`',
# 	'X' : '`Outlet Quality`',
# 	'T' : '`Inlet Temperature`',
# 	'H' : '`Inlet Subcooling`',
# 	'G' : '`Mass Flux`',
# }

# rank_condition = 'Tube Diameter'

# (L=[5.8, 6.05], G=[978, 1019], P=[14700, 14710], X=[0.378, 0.404], T=[36, 331])
# queryString = f"{condition['L']} >= 5.8 and {condition['L']} <= 6.05 and {condition['G']} >= 978 and {condition['G']} <= 1019 and {condition['P']} >= 14700 and {condition['P']} <= 14710 and {condition['T']} >= 36 and {condition['T']} <= 331"

# print(f'condition = {queryString}')
# dataExcel = data.query(queryString)
# print(dataExcel)


def readDataIfExist(name) :
    pkl_name = input_path + name + '.pkl'

    exist = os.path.exists(pkl_name)
    if exist :
        fileData  = pd.read_pickle(pkl_name)
        return fileData

    # 读取xlsx 获取数据
    fileData = pd.read_excel(ori_file_path, sheet_name=name, engine='openpyxl')
    fileData.to_pickle(pkl_name) #格式另存
    return fileData

LUT_list = []
valueSlice = []

for sheet_name in sheet_names :
    sheetList = readDataIfExist(sheet_name)
    sd =  np.array(sheetList['Number']).reshape(-1)
    if sd[0] == '-' :
        sd = np.delete(sd, 0)
    filterData = data.loc[sd]
    print(f'-------------------------------{sheet_name}-----------------------------')
    LUT_list = output_indicator("LUT vs Real", data['CHF LUT'], filterData['CHF'])
    aiList = output_indicator("AI vs Real", data['AI CHF'], filterData['CHF'])
    valueSlice.append(aiList)

sliceLUTMean = np.array(LUT_list).mean()
# 计算每一列的平均值，‌设置axis=0
sliceAIMean = np.mean(valueSlice, axis=1)
print(f'sliceLUTMean : {sliceLUTMean}  sliceAIMean {sliceAIMean}')  # 输出每一列的平均值



print(f'-------------------------------ALL-----------------------------')
LUT_list = output_indicator("LUT vs Real", data['CHF LUT'], data['CHF'])
aiList = output_indicator("AI vs Real", data['AI CHF'], data['CHF'])

sliceLUTMean = np.array(LUT_list).mean()
sliceAIMean = np.array(aiList).mean()
print(f'ALL LUTMean : {sliceLUTMean}  ALL AIMean {sliceAIMean}')  # 输出每一列的平均值



