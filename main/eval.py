import itertools
import numpy as np
import subprocess
import sys
import pandas as pd
import os

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
    result = [ rmspe, mape , nrmse_mean, EQ2]
    return [float(element) for element in result]

def readDataIfExist(name, input_path = '../input/',ori_file_path = '../data/slice_real_value (copy).xlsx') :
    pkl_name = input_path + name + '.pkl'

    exist = os.path.exists(pkl_name)
    if exist :
        fileData  = pd.read_pickle(pkl_name)
        return fileData

    # 读取xlsx 获取数据
    fileData = pd.read_excel(ori_file_path, sheet_name=name, engine='openpyxl')
    fileData.to_pickle(pkl_name) #格式另存
    return fileData


def eval(file_path) :
    data_file_path = file_path
    ori_file_path = '../data/slice_real_value (copy).xlsx'
    sheet_names = ['slice01','slice02','slice03','slice04','slice05','slice06','slice07','slice08','slice09','slice10']
    input_path = '../input/'

    data = pd.read_csv(data_file_path)
    data.drop(data.index[0], inplace=True)
    data = data.astype(float)

    LUT_list = []
    valueSlice = []

    remData = {}
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
        remData[sheet_name + '-LUT'] = list(LUT_list)
        remData[sheet_name + '-AI'] = list(aiList)

    sliceLUTMean = np.array(LUT_list).mean()
    # 计算每一列的平均值，‌设置axis=0
    sliceAIMean = np.mean(valueSlice, axis=1)
    print(f'sliceLUTMean : {sliceLUTMean}  sliceAIMean {sliceAIMean}')  # 输出每一列的平均值

    print(f'-------------------------------ALL-----------------------------')
    LUT_list = output_indicator("LUT vs Real", data['CHF LUT'], data['CHF'])
    aiList = output_indicator("AI vs Real", data['AI CHF'], data['CHF'])

    remData['ALL-LUT'] = LUT_list
    remData['ALL-AI'] = aiList

    sliceLUTMean = np.array(LUT_list).mean()
    sliceAIMean = np.array(aiList).mean()

    remData['Mean-ALL-LUT'] = list(LUT_list)
    remData['MeanALL-AI'] = list(aiList)
    print(f'ALL LUTMean : {sliceLUTMean}  ALL AIMean {sliceAIMean}')  # 输出每一列的平均值

    remData['Mean-value-ALL-LUT'] = sliceLUTMean
    remData['Mean-value-ALL-AI'] = sliceAIMean
    return remData


if __name__ == "__main__":
    a = eval('../output/AIData_ALL_model_transformer_findbest.csv')
    print(a)