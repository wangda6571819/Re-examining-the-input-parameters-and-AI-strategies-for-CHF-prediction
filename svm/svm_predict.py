from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Load the data
file_path = '../output/AIData_ALL_model_Transformer_110.csv'
data = pd.read_csv(file_path)

# Clean the data by removing the first row which contains units
data = data.iloc[1:].astype(float)

# Prepare the dataset
features_group_1 = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality']
features_group_2 = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Inlet Subcooling']
target = 'CHF'

X1_full = data[features_group_1]
X2_full = data[features_group_2]
y_full = data[target]

# Normalize the features
scaler = StandardScaler()
X1_full = scaler.fit_transform(X1_full)
X2_full = scaler.fit_transform(X2_full)

# Split the data into training and testing sets (80% train, 20% test)
X1_train_full, X1_test_full, y_train_full, y_test_full = train_test_split(X1_full, y_full, test_size=0.2, random_state=42)
X2_train_full, X2_test_full, _, _ = train_test_split(X2_full, y_full, test_size=0.2, random_state=42)

# Function to calculate advanced error metrics
def calculate_advanced_errors(y_true, y_pred):
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    nrmse = np.sqrt(mean_squared_error(y_true, y_pred)) / y_true.mean()
    q2 = 1 - r2_score(y_true, y_pred)
    od = np.mean([rmspe, mape, nrmse, q2])
    return rmspe, mape, nrmse, q2, od


# Train and predict with SVM using parallel processing
def train_and_evaluate(X_train, y_train, X_test):
    svm_model = SVR(kernel='linear')
    svm_model.fit(X_train, y_train)
    y_pred_test = svm_model.predict(X_test)
    return y_pred_test,svm_model

# Running SVM models in parallel
# y_pred_test_1_simplified = Parallel(n_jobs=-1)(delayed(train_and_evaluate)(X1_train_full, y_train_full, X1_test_full) for _ in range(1))
# y_pred_test_2_simplified= Parallel(n_jobs=-1)(delayed(train_and_evaluate)(X2_train_full, y_train_full, X2_test_full) for _ in range(1))
# print(y_pred_test_2_simplified)
# y_pred_test_2_simplified,svm_model = y_pred_test_2_simplified[0]

# print('执行完成')
# print(y_pred_test_2_simplified)
# Calculate advanced metrics for Group 1 and Group 2
# rmspe_test_1_simplified, mape_test_1_simplified, nrmse_test_1_simplified, q2_test_1_simplified, od_test_1_simplified = calculate_advanced_errors(y_test_full, y_pred_test_1_simplified[0])
# rmspe_test_2_simplified, mape_test_2_simplified, nrmse_test_2_simplified, q2_test_2_simplified, od_test_2_simplified = calculate_advanced_errors(y_test_full, y_pred_test_2_simplified)

# Output results
# print(rmspe_test_1_simplified, mape_test_1_simplified, nrmse_test_1_simplified, q2_test_1_simplified, od_test_1_simplified)
# print(rmspe_test_2_simplified, mape_test_2_simplified, nrmse_test_2_simplified, q2_test_2_simplified, od_test_2_simplified)



print('data dealing')

# 创建一个空的DataFrame
slice_sheet_name = ['slice01','slice02','slice03','slice04','slice05']
# 创建一个空的DataFrame
df_slice01 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])

# 设定 单值变动其他值不变
slice1D = np.arange(0.002, 0.017, 0.001)
df_slice01['Tube Diameter'] = slice1D
df_slice01['Heated Length'] = 2
df_slice01['Pressure'] = 10000
df_slice01['Mass Flux'] = 1550
df_slice01['Outlet Quality'] = [0.334] * len(slice1D)
df_slice01['Inlet Subcooling'] = 451
df_slice01['Inlet Temperature'] = 221

# 创建一个空的DataFrame
df_slice02 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])
# 设定 单值变动其他值不变
slice2D = np.arange(0, 21, 1)
df_slice02['Tube Diameter'] = [0.008] * len(slice2D)
df_slice02['Heated Length'] = slice2D
df_slice02['Pressure'] = 9841
df_slice02['Mass Flux'] = 1660
df_slice02['Outlet Quality'] = [0.327] * len(slice2D)
df_slice02['Inlet Subcooling'] = 448
df_slice02['Inlet Temperature'] = 221


# 创建一个空的DataFrame
df_slice03 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])
# 设定 单值变动其他值不变
slice3d = np.arange(0, 22000, 1500)
df_slice03['Tube Diameter'] = [0.008] * len(slice3d)
df_slice03['Heated Length'] = [1.9] * len(slice3d)
df_slice03['Pressure'] = slice3d
df_slice03['Mass Flux'] = 1544
df_slice03['Outlet Quality'] = [0.32] * len(slice3d)
df_slice03['Inlet Subcooling'] = 438
df_slice03['Inlet Temperature'] = 234


# 创建一个空的DataFrame
df_slice04 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])
# 设定 单值变动其他值不变
slice4d = np.arange(0, 8400, 400)
df_slice04['Tube Diameter'] = [0.008] * len(slice4d)
df_slice04['Heated Length'] = [1.76] * len(slice4d)
df_slice04['Pressure'] = 9839
df_slice04['Mass Flux'] = slice4d
df_slice04['Outlet Quality'] = [0.224] * len(slice4d)
df_slice04['Inlet Subcooling'] = 436
df_slice04['Inlet Temperature'] = 223


# 创建一个空的DataFrame
df_slice05 = pd.DataFrame(columns=['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Outlet Quality', 'Inlet Subcooling', 'Inlet Temperature'])
# 设定 单值变动其他值不变
slice5d = np.arange(0, 1500, 100)
df_slice05['Tube Diameter'] = [0.008] * len(slice5d)
df_slice05['Heated Length'] = [1.92] * len(slice5d)
df_slice05['Pressure'] = 9811
df_slice05['Mass Flux'] = 1574
df_slice05['Outlet Quality'] = [0.347] * len(slice5d)
df_slice05['Inlet Subcooling'] = slice5d
df_slice05['Inlet Temperature'] = 207


depend_items = {
'slice01' : {'key' : 'Tube Diameter' , 'df' : df_slice01},
'slice02' : {'key' : 'Heated Length' , 'df' : df_slice02},
'slice03': { 'key' : 'Pressure' , 'df' : df_slice03},
'slice04': { 'key' : 'Mass Flux', 'df' : df_slice04},
'slice05': { 'key' : 'Inlet Subcooling', 'df' : df_slice05},
}


svm_model = SVR(kernel='linear')
svm_model.fit(X2_train_full, y_train_full)

print(' start predict')
for slice_name in slice_sheet_name :
    new_path_data = f'../output/predict/slice_data_{slice_name}.csv'
    fdf = pd.read_csv(new_path_data)


    depend_item = depend_items[slice_name]
    slice_parameter_name = depend_item["key"]

    pdData = depend_item['df']

    prData = pdData[features_group_2]
    prData = scaler.fit_transform(prData)
    predictData = svm_model.predict(prData)
    print('-------------------')
    print(predictData)

    pdData[f'701 AI Data'] = predictData
    fdf[f'701 AI Data'] = predictData

    fdf.to_csv(new_path_data, index=False)




