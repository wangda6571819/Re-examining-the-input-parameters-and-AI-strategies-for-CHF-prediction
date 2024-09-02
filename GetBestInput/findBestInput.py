from transformer import doTrain
from eval import eval
import itertools
import numpy as np
import subprocess
import sys
import pandas as pd
import os

# 所有的参数信息
feature_columns_all = ['Tube Diameter' ,'Heated Length', 'Enthalpy value of water', 'Enthalpy value of air' ,'Enthalpy Temperature','Pressure', 'Mass Flux', 'Inlet Subcooling' ,'Inlet Temperature','Outlet Quality']
# 保存中间训练 feature loss 
feature_train_data_file = './data_feature.csv'
# 模型输出地址
model_path = '../model/Transformer_best_model_findBest.pth'
# 全量预测输出路径
output_path = '../output/AIData_ALL_model_transformer_findbest.csv'

# full time train
full_train_data_file = './data_-full_feature.csv'


#最小的参数个数
min_column_count = 1
# 寻找最佳input Epochs 次数
fetchBestEpochsCount = 120
# 模型 Epochs 次数  
trainBestEpochsCount = 2000
# get top 10
top_count = 10


def find_combinations(arr, min_elements):
    result = set()  # 使用集合来避免重复
    for r in range(min_elements, len(arr) + 1):
        for combo in itertools.combinations(arr, r):
            sorted_combo = tuple(sorted(combo))  # 确保组合中的元素顺序一致
            result.add(sorted_combo)
    return result


# 找到所有大于等于4个的组合
allCombinas = find_combinations(feature_columns_all, min_column_count)
allCombinas = [list(combo) for combo in allCombinas]

#输出结果
print(f'总共有{len(allCombinas)}种组合')
# print(allCombinas)


def predictAndEval(feature_columns,model_path,output_path):
    print('开始走预测流程')
    # 要执行的Python脚本文件名
    script_filename = './slice.py'
    # 将参数转成string
    feature_columns_string = (',').join(feature_columns)
    # 要传入的参数列表   features model_path output_path
    arguments = ['--featureColumns',feature_columns_string,'--modelPath', model_path ,'--outputPath', output_path]
    # 使用subprocess.run来执行脚本并传入参数
    print(f'执行slice.py : {[sys.executable, script_filename] + arguments}')
    result = subprocess.run([sys.executable, script_filename] + arguments, capture_output=True, text=True)
    # 打印执行结果
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")
    ######  开始准备走评估
    print(' 评估')
    out = eval(output_path)
    return out

bestLoss = 1
best_feature_columns = ['Enthalpy value of water', 'Heated Length', 'Inlet Subcooling', 'Inlet Temperature', 'Mass Flux', 'Outlet Quality', 'Pressure', 'Tube Diameter'] 
data_feature = []

header = ['feature_columns','best_val_loss','train_losses','val_losses']
eval_header = ['slice01','slice02','slice03','slice04','slice05','slice06','slice07','slice08','slice09','slice10','ALL','Mean-ALL','Mean-value-ALL']
end = ['-LUT','-AI']
append_header = [[x + y for x in eval_header] for y in end]
append_header = [item for sublist in append_header for item in sublist]

header = header + append_header
print(header)

df = pd.DataFrame(index=header)
df.drop(df.columns, axis=1, inplace=True)

for feature_columns_tuple in allCombinas:
    feature_columns = [x for x in feature_columns_tuple]
    print(f'feature_columns : {feature_columns} start train')
    train_losses, val_losses, best_val_loss = doTrain(feature_columns = feature_columns ,epochsCount = fetchBestEpochsCount)
    print(f'feature_columns : {feature_columns}  best loss: {best_val_loss}')

    dataAList = {'feature_columns' : feature_columns, 'best_val_loss' :str(best_val_loss),'train_losses' :train_losses,'val_losses' : val_losses}
    pridict = predictAndEval(feature_columns,model_path,output_path)

    dataAList.update(pridict)

    # 添加新行
    new_row = pd.Series(dataAList)
    df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    df = df.sort_values(by='Mean-value-ALL-AI')
    # 保存数组到文件
    # np.savetxt(feature_train_data_file,data_feature_array,delimiter=",",header ='',comments="",fmt = '%s')
    df.to_csv(feature_train_data_file, index=True)

top_ten_df = df.head(top_count)


print(f'best ten 组合是: {top_ten_df} ')

df_2000 = pd.DataFrame(index=header)
df_2000.drop(df_2000.columns, axis=1, inplace=True)

for index, row in top_ten_df.iterrows():

    # 全量预测输出路径
    output_path_best = f'../output/AIData_ALL_model_transformer_findbest_{index}.csv'
    model_path_best = f'../model/Transformer_best_model_findBest_{index}.pth'

    print(row["feature_columns"], row["best_val_loss"])
    feature_columns =  row["feature_columns"]
    train_losses, val_losses, best_val_loss = doTrain(feature_columns = feature_columns ,epochsCount = trainBestEpochsCount,model_path= model_path_best)
    print(f'feature_columns : {feature_columns}  best loss: { best_val_loss}')
    dataAList = {'feature_columns' : feature_columns, 'best_val_loss' :str(best_val_loss),'train_losses' :train_losses,'val_losses' : val_losses}
    pridict = predictAndEval(feature_columns,model_path_best,output_path_best)

    dataAList.update(pridict)

    # 添加新行
    new_row = pd.Series(dataAList)
    df_2000 = pd.concat([df_2000, new_row.to_frame().T], ignore_index=True)

    df_2000 = df_2000.sort_values(by='Mean-value-ALL-AI')

    df_2000.to_csv(full_train_data_file, index=True)