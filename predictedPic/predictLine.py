import itertools
import numpy as np
import subprocess
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from iapws import IAPWS97

from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.append('..')

# from transformer.transformer import doTrain as TransformerTrain
# from mamba.mamba_train import doTrain as MambaTrain
# from TCN.TCN_train import doTrain as TCNTrain

from transformer.transformer import getModel as TransformerTrain
from mamba.mamba_train import getModel as MambaTrain
from TCN.TCN_train import getModel as TCNTrain

import torch
import torch.nn as nn
import torch.nn.functional as F


epochs = 2000
# 测试列表
trainList = [
 # 110 6-1
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water','Enthalpy value of air' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '110', "trainMethod": TransformerTrain, 'model_name': 'Transformer','color': np.random.rand(3)},
# 111 5-1
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Pressure' , 'Mass Flux', 'Inlet Subcooling'], "callIndex" : '111', "trainMethod": TransformerTrain ,'model_name': 'Transformer','color': np.random.rand(3)},
# # 112 4-1
{ 'epochsCount' : epochs , 'feature_columns' : ['Pressure', 'Mass Flux', 'Inlet Subcooling','H/D'], "callIndex" : '112', "trainMethod": TransformerTrain ,'model_name': 'Transformer','color': np.random.rand(3)},
# # 113 5w -1
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length' ,'Enthalpy value of water', 'Mass Flux', 'Inlet Subcooling'], "callIndex" : '113', "trainMethod": TransformerTrain ,'model_name': 'Transformer','color': np.random.rand(3)},
# # 114 5a - 1
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length' ,'Enthalpy value of air', 'Mass Flux', 'Inlet Subcooling'], "callIndex" : '114', "trainMethod": TransformerTrain , 'model_name': 'Transformer','color': np.random.rand(3)},
# # 115 8-1
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' ,'Enthalpy Temperature', 'Mass Flux', 'Inlet Subcooling' ,'Inlet Temperature'], "callIndex" : '115', "trainMethod": TransformerTrain , 'model_name': 'Transformer','color': np.random.rand(3)},
# # 116 9-1
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' , 'Pressure', 'Mass Flux', 'Inlet Subcooling' ,'Inlet Temperature'], "callIndex" : '116', "trainMethod": TransformerTrain, 'model_name': 'Transformer','color': np.random.rand(3)},
# # 117 6+a
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' , 'Enthalpy Temperature' , 'Mass Flux', 'Inlet Temperature'], "callIndex" : '117', 'trainMethod': TransformerTrain , 'model_name': 'Transformer','color': np.random.rand(3)},
# # 118 6+b
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' ,'Enthalpy Temperature' , 'Mass Flux', 'Inlet Subcooling'], "callIndex" : '118', 'trainMethod': TransformerTrain , 'model_name': 'Transformer','color': np.random.rand(3)},

# # 119 LDPXG
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality'], 'callIndex' : '119', 'trainMethod': TransformerTrain, 'model_name': 'Transformer','color': np.random.rand(3)},
# # 120 DPXG
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Pressure', 'Mass Flux', 'Outlet Quality'], 'callIndex' : '120', 'trainMethod': TransformerTrain, 'model_name': 'Transformer','color': np.random.rand(3)},


# # 200 6-1 mamba 
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water','Enthalpy value of air' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '200', 'trainMethod': MambaTrain, 'model_name': 'Mamba','color': np.random.rand(3)},
# 201 LDPXG
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Pressure' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '201', 'trainMethod': MambaTrain, 'model_name': 'Mamba','color': np.random.rand(3)},


# 300 6-1 tcn 
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water','Enthalpy value of air' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '300', 'trainMethod': TCNTrain, 'model_name': 'TCN','color': np.random.rand(3)},
# 301
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Pressure' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '301', 'trainMethod': TCNTrain, 'model_name': 'TCN','color': np.random.rand(3)},
]


slice_sheet_name = ['slice01','slice02','slice03','slice04','slice05','slice06','slice07','slice08','slice09','slice10','slice11','slice12']

# 创建一个空的DataFrame
df_slice01 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])

# 设定 单值变动其他值不变
slice1D = np.arange(0.002, 0.017, 0.001)
df_slice01['Tube Diameter'] = slice1D
df_slice01['Heated Length'] = 6
df_slice01['Pressure'] = 14710
df_slice01['Mass Flux'] = 1000
df_slice01['Outlet Quality'] = [0.53] * len(slice1D)
df_slice01['Inlet Subcooling'] = 435
df_slice01['Inlet Temperature'] = 258


# 创建一个空的DataFrame
df_slice02 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])
df_slice02['Tube Diameter'] = slice1D
df_slice02['Heated Length'] = 6
df_slice02['Pressure'] = 9800
df_slice02['Mass Flux'] = 1000
df_slice02['Outlet Quality'] = [0.53] * len(slice1D)
df_slice02['Inlet Subcooling'] = 650
df_slice02['Inlet Temperature'] = 173


# 创建一个空的DataFrame
df_slice03 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])

# 设定 单值变动其他值不变
slice3d = np.arange(0, 21, 1)
df_slice03['Tube Diameter'] = [0.008] * len(slice3d)
df_slice03['Heated Length'] = slice3d
df_slice03['Pressure'] = 9800
df_slice03['Mass Flux'] = 1005
df_slice03['Outlet Quality'] = [0.58] * len(slice3d)
df_slice03['Inlet Subcooling'] = 500
df_slice03['Inlet Temperature'] = 207


# 创建一个空的DataFrame
df_slice04 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])
df_slice04['Tube Diameter'] = [0.0081] * len(slice3d)
df_slice04['Heated Length'] = slice3d
df_slice04['Pressure'] = 2000
df_slice04['Mass Flux'] = 751
df_slice04['Outlet Quality'] = [0.76] * len(slice3d)
df_slice04['Inlet Subcooling'] = 219
df_slice04['Inlet Temperature'] = 162


# 创建一个空的DataFrame
df_slice05 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])
# 设定 单值变动其他值不变
slice5d = np.arange(0, 22000, 1500)
df_slice05['Tube Diameter'] = [0.008] * len(slice5d)
df_slice05['Heated Length'] = 1
df_slice05['Pressure'] = slice5d
df_slice05['Mass Flux'] = 2000
df_slice05['Outlet Quality'] = [0.14] * len(slice5d)
df_slice05['Inlet Subcooling'] = 485
df_slice05['Inlet Temperature'] = 210

df_slice06 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])
df_slice06['Tube Diameter'] = [0.0133] * len(slice5d)
df_slice06['Heated Length'] = [3.658] * len(slice5d)
df_slice06['Pressure'] = slice5d
df_slice06['Mass Flux'] = 2038
df_slice06['Outlet Quality'] = [0.377] * len(slice5d)
df_slice06['Inlet Subcooling'] = 164
df_slice06['Inlet Temperature'] = 254


# 创建一个空的DataFrame
df_slice07 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])
# 设定 单值变动其他值不变
slice7d = np.arange(0, 8400, 400)
df_slice07['Tube Diameter'] = [0.008] * len(slice7d)
df_slice07['Heated Length'] = [1.57] * len(slice7d)
df_slice07['Pressure'] = 12750
df_slice07['Mass Flux'] = slice7d
df_slice07['Outlet Quality'] = [0.142] * len(slice7d)
df_slice07['Inlet Subcooling'] = 428
df_slice07['Inlet Temperature'] = 247

df_slice08 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])
df_slice08['Tube Diameter'] = [0.01] * len(slice7d)
df_slice08['Heated Length'] = 4.966
df_slice08['Pressure'] = 16000
df_slice08['Mass Flux'] = slice7d
df_slice08['Outlet Quality'] = [0.344] * len(slice7d)
df_slice08['Inlet Subcooling'] = 154
df_slice08['Inlet Temperature'] = 327

# # 创建一个空的DataFrame
df_slice09 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])
# 设定 单值变动其他值不变
slice9d = np.arange(-0.5, 1.1, 0.1)
df_slice09['Tube Diameter'] = [0.0081] * len(slice9d)
df_slice09['Heated Length'] = [1.94] * len(slice9d)
df_slice09['Pressure'] = 9831
df_slice09['Mass Flux'] = 1519
df_slice09['Outlet Quality'] = slice9d
df_slice09['Inlet Subcooling'] = 444
df_slice09['Inlet Temperature'] = 218

df_slice10 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])
df_slice10['Tube Diameter'] = [0.008] * len(slice9d)
df_slice10['Heated Length'] = 1
df_slice10['Pressure'] = 17650
df_slice10['Mass Flux'] = 2003
df_slice10['Outlet Quality'] = slice9d
df_slice10['Inlet Subcooling'] = 824
df_slice10['Inlet Temperature'] = 202


# 创建一个空的DataFrame
df_slice11 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])
# 设定 单值变动其他值不变
# slice9d = np.arange(-0.5, 1.1, 0.1)
slice11d = np.arange(0, 1600, 100)
df_slice11['Tube Diameter'] = [0.0081] * len(slice11d)
df_slice11['Heated Length'] = [1.94] * len(slice11d)
df_slice11['Pressure'] = 9831
df_slice11['Mass Flux'] = 1519
df_slice11['Outlet Quality'] = [0.371] * len(slice11d)
df_slice11['Inlet Subcooling'] = slice11d
df_slice11['Inlet Temperature'] = 218

df_slice12 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])
df_slice12['Tube Diameter'] = [0.008] * len(slice11d)
df_slice12['Heated Length'] = 1
df_slice12['Pressure'] = 17650
df_slice12['Mass Flux'] = [2000] * len(slice11d)
df_slice12['Outlet Quality'] = [-0.14] * len(slice11d)
df_slice12['Inlet Subcooling'] = slice11d
df_slice12['Inlet Temperature'] = 202

depend_items = {
'slice01' : {'key' : 'Tube Diameter' , 'df' : df_slice01},
'slice02' : {'key' : 'Tube Diameter' , 'df' : df_slice02},
'slice03' : {'key' : 'Heated Length' , 'df' : df_slice03},
'slice04' : {'key' : 'Heated Length' , 'df' : df_slice04},
'slice05': { 'key' : 'Pressure' , 'df' : df_slice05},
'slice06': { 'key' : 'Pressure', 'df' : df_slice06},
'slice07': { 'key' : 'Mass Flux', 'df' : df_slice07},
'slice08': { 'key' : 'Mass Flux', 'df' : df_slice08},
'slice09': { 'key' : 'Outlet Quality', 'df' : df_slice09},
'slice10': { 'key' : 'Outlet Quality', 'df' : df_slice10},
'slice11': { 'key' : 'Inlet Subcooling', 'df' : df_slice11},
'slice12': { 'key' : 'Inlet Subcooling', 'df' : df_slice12},
}

data_file_path = '../data/slice_real_value (copy).xlsx'
public_sheet_name = 'chf_public_LUT'
input_path = '../input/'
public_data_path = './public_data.csv';
out_put_path = './pic/'

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


def has_file(directory):
    # 列出目录中的所有文件
    files = os.listdir(directory)
    # 检查是否有.csv文件
    for file in files:
        if file.endswith('.csv'):
            return True
    return False

def predictVal(featureColumns, model_path,model, pdData) :

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    public_sheet_name = 'chf_public_LUT'
    allData = readDataIfExist(public_sheet_name)

    allData = allData.drop(0)
    # data_to_transform : 从第三列开始的所有数据列
    data_to_transform = allData.iloc[:, 2:]
    # data_to_transform = data_to_transform.drop(['CHF','CHF LUT'], axis=1)

    data_to_transform['Enthalpy value of water'] = data_to_transform.apply(lambda x : calculateEnthalpy(x['Pressure'])[0],axis=1)
    data_to_transform['Enthalpy value of air'] = data_to_transform.apply(lambda x : calculateEnthalpy(x['Pressure'])[1],axis=1)
    data_to_transform['Enthalpy Temperature'] = data_to_transform.apply(lambda x : calculateEnthalpyTemperature(x['Pressure']),axis=1)
    data_to_transform['H/D'] = data_to_transform.apply(lambda x : calculateHD(x),axis=1)


    # data_to_transform = data_to_transform.reindex(columns=featureColumns)

    # 初始化StandardScaler的字典来存储每列的归一化容器
    scalers = {}
    # 对每列进行归一化
    normalized_columns = {}
    for column in data_to_transform.columns:
        scaler = StandardScaler()
        normalized_columns[column] = scaler.fit_transform(data_to_transform[[column]].values).flatten()
        scalers[column] = scaler

    # 创建新字典 来存标准化字典
    new_scalers = {key: data[key] for key in featureColumns if key in data}
 
    # 应用预先定义的scalers对slice_data_2d的每一列进行归一化
    normalized_slice_data_2d =  pdData[featureColumns].values
    # 遍历每一列，应用相应的归一化容器
    for i, column_name in enumerate(new_scalers):
        normalized_slice_data_2d[:, i] = scalers[column_name].transform(normalized_slice_data_2d[:, i].reshape(-1, 1)).flatten()
    

    # 加载保存的模型权重
    pth = torch.load(model_path,map_location=device)
    model.load_state_dict(pth)
    model.eval()  # 设置为评估模式

    normalized_slice_data_2d = normalized_slice_data_2d.astype(float)
    input_data = torch.tensor(normalized_slice_data_2d, dtype=torch.float)
    input_data = input_data.unsqueeze(0)

    print('-------- 开始预测值-----')
    with torch.no_grad():
        data = input_data.to(device)
        x = model(data)
        x = x.to(torch.device("cpu"))

    # 为了逆变换，需要确保它是二维的
    output = x.reshape(-1, 1)
    # 使用最后一列的StandardScaler实例进行逆变换
    output = scalers['CHF'].inverse_transform(output)
    output = np.clip(output, 50, 16339)
    # 转换为一维数组
    out_put_float = output.reshape(-1)
    print(out_put_float)
    return out_put_float

public_data = []
# 检查文件是否存在
if os.path.exists(public_data_path):
    print(f'文件 {public_data_path} 存在。')
    public_data  = pd.read_csv(public_data_path)
else:

    public_data = readDataIfExist(public_sheet_name)

    public_data = public_data.drop([0])

    public_data['Enthalpy value of water'] = public_data.apply(lambda x : calculateEnthalpy(x['Pressure'])[0],axis=1)
    public_data['Enthalpy value of air'] = public_data.apply(lambda x : calculateEnthalpy(x['Pressure'])[1],axis=1)
    public_data['Enthalpy Temperature'] = public_data.apply(lambda x : calculateEnthalpyTemperature(x['Pressure']),axis=1)
    public_data['H/D'] = public_data.apply(lambda x : calculateHD(x),axis=1)
    public_data.to_csv(public_data_path, index=False)

print(public_data)

# 先给准预测值加上 计算属性
for depend_item in depend_items :
    print(depend_item)
    items = depend_items[depend_item]

    data = items['df']
    print(data)
    data['Enthalpy value of water'] = data.apply(lambda x : calculateEnthalpy(x['Pressure'])[0],axis=1)
    data['Enthalpy value of air'] = data.apply(lambda x : calculateEnthalpy(x['Pressure'])[1],axis=1)
    data['Enthalpy Temperature'] = data.apply(lambda x : calculateEnthalpyTemperature(x['Pressure']),axis=1)
    data['H/D'] = data.apply(lambda x : calculateHD(x),axis=1)


for slice_name in slice_sheet_name :
    slice_sheet_data = readDataIfExist(slice_name)
    print(slice_sheet_data)

    slice_number_list =  np.array(slice_sheet_data['Number']).reshape(-1)
    if slice_number_list[0] == '-' :
        slice_number_list = np.delete(slice_number_list, 0)
    
    # 获取最后一列的数据，排除第一个值
    lut_data = slice_sheet_data.iloc[:, -1].values[1:].astype(float) / 1000


    depend_item = depend_items[slice_name]
    slice_parameter_name = depend_item["key"]

    slice_sheet_data_outIndex = slice_sheet_data.iloc[1:]
    slice_parameter =  slice_sheet_data_outIndex[slice_parameter_name]

    pdData = depend_item['df']


    plt.figure(figsize=(5,3),dpi=1000)
    # 绘制真实点
    plt.scatter(slice_sheet_data[slice_parameter_name], pd.to_numeric(slice_sheet_data['CHF']), color='purple', marker='D',  label='Real data', s=10)


    for trainItem in trainList :
        callIndex = trainItem['callIndex']
        model_name = trainItem['model_name']
        feature_columns = trainItem['feature_columns']
        color = trainItem['color']

        # 模型输出位置
        model_path = f'../model/{model_name}_best_model_{callIndex}.pth'
        # 全量预测输出路径
        output_path = f'../output/AIData_ALL_model_{model_name}_{callIndex}.csv'

        # slice_filter_list = slice_number_list.reindex(columns=feature_columns)
        # print(slice_filter_list)

        model = trainItem['trainMethod'](feature_columns = feature_columns)

        print(callIndex)
        predictData = predictVal(feature_columns,model_path, model,pdData)
        # plt.plot(sort_filer_data[slice_parameter_name], pd.to_numeric(sort_filer_data['CHF LUT']), marker='x', color='black', label='LUT data', linewidth=2, markersize=8)

        pdData[f'{callIndex} AI Data'] = predictData
        # 绘制AI点
        # plt.plot(pdData[slice_parameter_name], predictData, marker='d', color=color,  label=f'{callIndex}AI data', linewidth=1, markersize=2)
        # plt.show()

    pdData.to_csv(f'./slice_data_{slice_name}.csv', index=False)
    # plt.title(slice_parameter_name+' vs pre CHF', fontsize=14)
    # plt.xlabel(slice_parameter_name, fontsize=6)
    # plt.ylabel('Pre CHF [kW/m^2]', fontsize=6)

    # Enlarging the tick marks' font size
    # plt.xticks(fontsize=4)
    # plt.yticks(fontsize=4)
    # plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.05, 0.5), loc=6, borderaxespad=0)
    # plt.tight_layout(rect=[0, 0, 1, 1])
    # plt.show()

    # 确保文件夹路径存在，不存在则创建
    # if not os.path.exists(f'{out_put_path}/{slice_name}/'):
        # os.makedirs(f'{out_put_path}/{slice_name}/')
    # 保存图表
    # plt.savefig(f'{out_put_path}/{slice_name}/{slice_name}.png',bbox_inches='tight')

    # 关闭图表
    # plt.close()






