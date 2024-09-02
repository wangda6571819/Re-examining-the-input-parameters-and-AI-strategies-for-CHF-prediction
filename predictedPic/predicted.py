import itertools
import numpy as np
import subprocess
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from iapws import IAPWS97


sys.path.append('..')

from transformer.transformer import doTrain as TransformerTrain
from mamba.mamba_train import doTrain as MambaTrain
from TCN.TCN_train import doTrain as TCNTrain

epochs = 2000
# 测试列表
trainList = [
 # 110 6-1
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water','Enthalpy value of air' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '110', "trainMethod": TransformerTrain, 'model_name': 'Transformer'},
# # 111 5-1
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Pressure' , 'Mass Flux', 'Inlet Subcooling'], "callIndex" : '111', "trainMethod": TransformerTrain ,'model_name': 'Transformer'},
# # # 112 4-1
# { 'epochsCount' : epochs , 'feature_columns' : ['Pressure', 'Mass Flux', 'Inlet Subcooling','H/D'], "callIndex" : '112', "trainMethod": TransformerTrain ,'model_name': 'Transformer'},
# # # 113 5w -1
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length' ,'Enthalpy value of water', 'Mass Flux', 'Inlet Subcooling'], "callIndex" : '113', "trainMethod": TransformerTrain ,'model_name': 'Transformer'},
# # # 114 5a - 1
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length' ,'Enthalpy value of air', 'Mass Flux', 'Inlet Subcooling'], "callIndex" : '114', "trainMethod": TransformerTrain , 'model_name': 'Transformer'},
# # # 115 8-1
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' ,'Enthalpy Temperature', 'Mass Flux', 'Inlet Subcooling' ,'Inlet Temperature'], "callIndex" : '115', "trainMethod": TransformerTrain , 'model_name': 'Transformer'},
# # # 116 9-1
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' , 'Pressure', 'Mass Flux', 'Inlet Subcooling' ,'Inlet Temperature'], "callIndex" : '116', "trainMethod": TransformerTrain, 'model_name': 'Transformer'},
# # # 117 6+a
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' , 'Enthalpy Temperature' , 'Mass Flux', 'Inlet Subcooling' ,'Inlet Temperature'], "callIndex" : '117', 'trainMethod': TransformerTrain , 'model_name': 'Transformer'},
# # # 118 6+b
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' ,'Enthalpy Temperature' , 'Mass Flux', 'Inlet Subcooling'], "callIndex" : '118', 'trainMethod': TransformerTrain , 'model_name': 'Transformer'},

# # # 119 LDPXG
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality'], 'callIndex' : '119', 'trainMethod': TransformerTrain, 'model_name': 'Transformer'},
# # # 120 DPXG
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Pressure', 'Mass Flux', 'Outlet Quality'], 'callIndex' : '120', 'trainMethod': TransformerTrain, 'model_name': 'Transformer'},


# # # 200 6-1 mamba 
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Pressure' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '200', 'trainMethod': MambaTrain, 'model_name': 'Mamba'},
# # # 201 LDPXG
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Pressure' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '201', 'trainMethod': MambaTrain, 'model_name': 'Mamba'},


# # # 300 6-1 tcn 
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water','Enthalpy value of air' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '300', 'trainMethod': TCNTrain, 'model_name': 'TCN'},
# # 301
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Pressure' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '301', 'trainMethod': TCNTrain, 'model_name': 'TCN'},
]


slice_sheet_name = ['slice01','slice02','slice03','slice04','slice05','slice06','slice07','slice08','slice09','slice10']

depend_items = {
'slice01' : {'key' : 'Tube Diameter' },
'slice02' : {'key' : 'Tube Diameter' },
'slice03' : {'key' : 'Heated Length'},
'slice04' : {'key' : 'Heated Length'},
'slice05': { 'key' : 'Pressure'},
'slice06': { 'key' : 'Pressure'},
'slice07': { 'key' : 'Mass Flux'},
'slice08': { 'key' : 'Mass Flux'},
'slice09': { 'key' : 'Outlet Quality'},
'slice10': { 'key' : 'Outlet Quality'},
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


	for trainItem in trainList :
		callIndex = trainItem['callIndex']
		model_name = trainItem['model_name']
		feature_columns = trainItem['feature_columns']

		# 模型输出位置
		model_path = f'../model/{model_name}_best_model_{callIndex}.pth'
		# 全量预测输出路径
		output_path = f'../output/AIData_ALL_model_{model_name}_{callIndex}.csv'

		pre_data_all = pd.read_csv(output_path)
		filterData = pre_data_all.loc[slice_number_list]
		print(filterData)
		sort_filer_data = filterData.sort_values(by=slice_parameter_name)
		# slice_filter_list = slice_number_list.reindex(columns=feature_columns)
		# print(slice_filter_list)

		plt.figure()
		plt.plot(sort_filer_data[slice_parameter_name], pd.to_numeric(sort_filer_data['CHF LUT']), marker='x', color='black', label='LUT data', linewidth=2, markersize=8)

		plt.scatter(sort_filer_data[slice_parameter_name], pd.to_numeric(sort_filer_data['CHF']), color='purple', marker='D',  label='Real data')

		plt.plot(sort_filer_data[slice_parameter_name], pd.to_numeric(sort_filer_data['AI CHF']), marker='x', color='red',  label='AI data', linewidth=2, markersize=8)

		plt.title(slice_parameter_name+' vs pre CHF', fontsize=16)
		plt.xlabel(slice_parameter_name, fontsize=14)
		plt.ylabel('Pre CHF [kW/m^2]', fontsize=14)

		# Enlarging the tick marks' font size
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		plt.grid(True)
		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		plt.tight_layout(rect=[0, 0, 1, 1])
		# 添加图例
		plt.legend()
		# plt.show()

		# 确保文件夹路径存在，不存在则创建
		if not os.path.exists(f'{out_put_path}/{callIndex}/'):
		    os.makedirs(f'{out_put_path}/{callIndex}/')
		# 保存图表
		plt.savefig(f'{out_put_path}/{callIndex}/{model_name}_{callIndex}_{slice_parameter_name}_{slice_name}.png')

		# 关闭图表
		plt.close()






