
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

feature_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Inlet Subcooling']

def readDataIfExist(name) :
    input_path = '../input/'
    pkl_name = input_path + name + '.pkl'

    exist = os.path.exists(pkl_name)
    if exist :
        data  = pd.read_pickle(pkl_name)
        return data

    # 读取xlsx 获取数据
    data = pd.read_excel(data_file_path, sheet_name=name, engine='openpyxl')
    data.to_pickle(pkl_name) #格式另存
    return data

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

def calculateHD(row):
    hd = float(row['Heated Length']) / float(row['Tube Diameter'])
    return hd

def predictAndEval(featureColumns = ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' , 'Mass Flux', 'Inlet Subcooling'] , model_path = '../model/Transformer_best_model_119.pth' , output_path = '../output/AIData_ALL_model_transformer_119_6_1(1000).csv', model = '1') :
    

    pd.set_option('display.width', 50000) # 设置字符显示宽度
    pd.set_option('display.max_rows', None) # 设置显示最大行
    pd.set_option('display.max_columns', None) # 设置显示最大列，None为显示所有列


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slice_real_file_path = '../data/slice_real_value (copy).xlsx'

    data_file_path = '../data/slice_real_value (copy).xlsx'
    public_sheet_name = 'chf_public_LUT'
    sheet_name = 'slice02'
    input_path = '../input/'

    slice_parameter_name = 'Tube Diameter [m]'
    slice_parameter_name_1 = 'Tube Diameter'
    slice_parameter_index = 0
    input_features = len(featureColumns)
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

    data_to_transform['Enthalpy value of water'] = data_to_transform.apply(lambda x : calculateEnthalpy(x['Pressure'])[0],axis=1)
    data_to_transform['Enthalpy value of air'] = data_to_transform.apply(lambda x : calculateEnthalpy(x['Pressure'])[1],axis=1)
    data_to_transform['Enthalpy Temperature'] = data_to_transform.apply(lambda x : calculateEnthalpyTemperature(x['Pressure']),axis=1)
    data_to_transform['H/D'] = data_to_transform.apply(lambda x : calculateHD(x),axis=1)


    data_to_transform = data_to_transform.reindex(columns=featureColumns)

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
    normalized_slice_data_2d =  data_to_transform[featureColumns].values

    # 遍历每一列，应用相应的归一化容器
    for i, column_name in enumerate(scalers):
        normalized_slice_data_2d[:, i] = scalers[column_name].transform(normalized_slice_data_2d[:, i].reshape(-1, 1)).flatten()

    print(normalized_slice_data_2d)


    # 加载保存的模型权重
    pth = torch.load(model_path,map_location=device)
    model.load_state_dict(pth)
    model.eval()  # 设置为评估模式

    # ————————————————————————————————————————————————————————————得到结果————————————————————————————————————————————————————————————

    normalized_slice_data_2d = normalized_slice_data_2d.astype(float)
    input_data = torch.tensor(normalized_slice_data_2d, dtype=torch.float)
    input_data = input_data.unsqueeze(0)

    print('开始分段预测结果')
    list_input_data = torch.chunk(input_data, dim = 1 , chunks = 10)
    

    output = np.array([])
    for idx, iData in enumerate(list_input_data) :

        with torch.no_grad():
            data = iData.to(device)
            x = model(data)
            x = x.to(torch.device("cpu"))
            output = np.append(output, x)


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


