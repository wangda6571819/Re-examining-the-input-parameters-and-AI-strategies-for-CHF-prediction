import itertools
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

data_file_path = '../data/slice_real_value (copy).xlsx'

ai_data_list = ['121', '202', '302', '401', '501', '601', '701', '801']
# Define fixed colors, markers, and line styles for each AI data line
ai_data_styles = {
    '121': {'color': 'red', 'marker': 'o', 'linestyle': '-'},
    '202': {'color': 'blue', 'marker': 's', 'linestyle': '--'},
    '302': {'color': 'green', 'marker': '^', 'linestyle': '-.'},
    '401': {'color': 'orange', 'marker': 'v', 'linestyle': ':'},
    '501': {'color': 'purple', 'marker': 'p', 'linestyle': '-'},
    '601': {'color': 'navy', 'marker': '3', 'linestyle': '--'},
    '701': {'color': 'pink', 'marker': 'h', 'linestyle': '-.'},
    '801': {'color': 'cyan', 'marker': 'x', 'linestyle': ':'}
}

depend_items = {
    'slice01': {'key': 'Tube Diameter', 'unit': 'm', 'realData': 'slice01'},
    'slice02': {'key': 'Heated Length', 'unit': 'm', 'realData': 'slice02'},
    'slice03': {'key': 'Pressure', 'unit': 'kg/m^2/s', 'realData': 'slice03'},
    'slice04': {'key': 'Mass Flux', 'unit': '-', 'realData': 'slice04'},
    'slice05': {'key': 'Inlet Subcooling', 'unit': 'kJ/kg', 'realData': 'slice05'},
}

def readDataIfExist(name):
    input_path = '../input/'
    pkl_name = input_path + name + '.pkl'

    exist = os.path.exists(pkl_name)
    if exist:
        data = pd.read_pickle(pkl_name)
        return data

    data = pd.read_excel(data_file_path, sheet_name=name, engine='openpyxl')
    data.to_pickle(pkl_name)
    return data

for slice_name in depend_items.keys():
    depend_item = depend_items[slice_name]

    slice_parameter_name = depend_item['key']
    unit = depend_item['unit']

    predict_path = f'../output/predict/slice_data_{slice_name}.csv'
    predict_data = pd.read_csv(predict_path)

    real_data_slice_name = depend_item['realData']
    slice_sheet_data = pd.read_csv(f'../data/slice_data/{real_data_slice_name}.csv')

    plt.figure(figsize=(6, 4), dpi=300)
    plt.scatter(slice_sheet_data[slice_parameter_name], pd.to_numeric(slice_sheet_data['CHF']), color='black', marker='D', label='Real data', s=40)

    for model_index in ai_data_list:
        style = ai_data_styles[model_index]
        cloumn_name = f'{model_index} AI Data'
        
        if model_index == '501':
            plt.plot(predict_data[slice_parameter_name], predict_data[cloumn_name], 
                     marker=style['marker'], color=style['color'], linestyle=style['linestyle'], 
                     label=f'{model_index} AI data', linewidth=1, markersize=6, markerfacecolor='none')
        else:
            plt.plot(predict_data[slice_parameter_name], predict_data[cloumn_name], 
                     marker=style['marker'], color=style['color'], linestyle=style['linestyle'], 
                     label=f'{model_index} AI data', linewidth=1, markersize=4)

    # Adding LUT values with fixed style
    plt.plot(predict_data[slice_parameter_name], predict_data['LUT Data'], marker='d', color='grey', linestyle='-', label='LUT data', linewidth=1, markersize=4)

    plt.title(slice_parameter_name + ' vs pre CHF', fontsize=12)
    plt.xlabel(slice_parameter_name + ' (' + unit + ') ', fontsize=12)
    plt.ylabel('Pre CHF [kW/m^2]', fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend(loc=0, borderaxespad=0, fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig(f'../output/pic/{slice_name}.png', bbox_inches='tight')
    plt.close()
