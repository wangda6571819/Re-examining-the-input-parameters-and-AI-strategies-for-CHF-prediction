
import itertools
import numpy as np
import subprocess
import sys
import pandas as pd
import os
sys.path.append('..')

from transformer.transformer import doTrain as TransformerTrain
from mamba.mamba_train import doTrain as MambaTrain
from TCN.TCN_train import doTrain as TCNTrain

from eval import eval

from slice import predictAndEval
import datetime

# 训练次数
epochs = 2000

# 保存中间训练 feature loss 
feature_train_data_file = './data_feature.csv'


# 测试列表
trainList = [
 # 110 6-1
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water','Enthalpy value of air' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '110', "trainMethod": TransformerTrain, 'model_name': 'Transformer'},
# 111 5-1
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Pressure' , 'Mass Flux', 'Inlet Subcooling'], "callIndex" : '111', "trainMethod": TransformerTrain ,'model_name': 'Transformer'},
# # 112 4-1
# { 'epochsCount' : epochs , 'feature_columns' : ['Pressure', 'Mass Flux', 'Inlet Subcooling','H/D'], "callIndex" : '112', "trainMethod": TransformerTrain ,'model_name': 'Transformer'},
# # 113 5w -1
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length' ,'Enthalpy value of water', 'Mass Flux', 'Inlet Subcooling'], "callIndex" : '113', "trainMethod": TransformerTrain ,'model_name': 'Transformer'},
# # 114 5a - 1
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length' ,'Enthalpy value of air', 'Mass Flux', 'Inlet Subcooling'], "callIndex" : '114', "trainMethod": TransformerTrain , 'model_name': 'Transformer'},
# # 115 8-1
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' ,'Enthalpy Temperature', 'Mass Flux', 'Inlet Subcooling' ,'Inlet Temperature'], "callIndex" : '115', "trainMethod": TransformerTrain , 'model_name': 'Transformer'},
# # 116 9-1
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' , 'Pressure', 'Mass Flux', 'Inlet Subcooling' ,'Inlet Temperature'], "callIndex" : '116', "trainMethod": TransformerTrain, 'model_name': 'Transformer'},
# # 117 6+a
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' , 'Enthalpy Temperature' , 'Mass Flux', 'Inlet Subcooling' ,'Inlet Temperature'], "callIndex" : '117', 'trainMethod': TransformerTrain , 'model_name': 'Transformer'},
# # 118 6+b
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' ,'Enthalpy Temperature' , 'Mass Flux', 'Inlet Subcooling'], "callIndex" : '118', 'trainMethod': TransformerTrain , 'model_name': 'Transformer'},

# 119 LDPXG
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality'], 'callIndex' : '119', 'trainMethod': TransformerTrain, 'model_name': 'Transformer'},
# # 120 DPXG
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Pressure', 'Mass Flux', 'Outlet Quality'], 'callIndex' : '120', 'trainMethod': TransformerTrain, 'model_name': 'Transformer'},

# 121 
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Pressure' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '121', "trainMethod": TransformerTrain, 'model_name': 'Transformer'},

# 200 6-1 mamba 
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water','Enthalpy value of air' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '200', 'trainMethod': MambaTrain, 'model_name': 'Mamba'},
# 201  LDPXG
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality'], 'callIndex' : '201', 'trainMethod': MambaTrain, 'model_name': 'Mamba'},
# 202 LDPXG
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Pressure' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '202', 'trainMethod': MambaTrain, 'model_name': 'Mamba'},


# 300 6-1 tcn 
# { 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Enthalpy value of water','Enthalpy value of air' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '300', 'trainMethod': TCNTrain, 'model_name': 'TCN'},
# 302 LDPXG
{ 'epochsCount' : epochs , 'feature_columns' : ['Tube Diameter', 'Heated Length', 'Pressure' , 'Mass Flux', 'Inlet Subcooling'], 'callIndex' : '302', 'trainMethod': TCNTrain, 'model_name': 'TCN'},
]

header = ['callIndex','feature_columns','best_val_loss','train_losses','val_losses']
eval_header = ['slice01','slice02','slice03','slice04','slice05','slice06','slice07','slice08','slice09','slice10','ALL','Mean-ALL','Mean-value-ALL']
end = ['-LUT','-AI']
append_header = [[x + y for x in eval_header] for y in end]
append_header = [item for sublist in append_header for item in sublist]

header = header + append_header
print(header)

df = pd.DataFrame(index=header)
df.drop(df.columns, axis=1, inplace=True)

for trainItem in trainList :
	callIndex = trainItem['callIndex']
	feature_columns = trainItem['feature_columns']
	epochsCount = trainItem['epochsCount']
	model_name = trainItem['model_name']
	# 模型输出位置
	model_path = f'../model/{model_name}_best_model_{callIndex}.pth'
	# 全量预测输出路径
	output_path = f'../output/AIData_ALL_model_{model_name}_{callIndex}.csv'

	print(f'callIndex : {callIndex} feature_columns : {feature_columns} start train')
	train_losses, val_losses, best_val_loss, model = trainItem['trainMethod'](feature_columns = feature_columns ,epochsCount = epochsCount,callIndex = callIndex)

	dataAList = {'callIndex' : callIndex,'feature_columns' : feature_columns, 'best_val_loss' :best_val_loss,'train_losses' :train_losses,'val_losses' : val_losses ,'start_time': datetime.datetime.now()}
	pridict = predictAndEval(feature_columns, model_path, output_path, model)
	
	dataAList['start_time'] = datetime.datetime.now()

	######  开始准备走评估
	print('start to eval data')
	pridict = eval(output_path)

	dataAList.update(pridict)

    # 添加新行
	new_row = pd.Series(dataAList)
	df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
	
	# 不需要排序
	df = df.sort_values(by='Mean-value-ALL-AI')
    # 保存数组到文件
	df.to_csv(feature_train_data_file, index=True)



