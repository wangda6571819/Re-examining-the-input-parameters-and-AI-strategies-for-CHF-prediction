import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the data
file_path = '../output/AIData_ALL_model_Transformer_110.csv'  # Replace with the actual path to your CSV file
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

# Initialize Ridge Regression with default configurations
ridge_model = Ridge(random_state=42)

# Train and predict with Group 1 features
ridge_model.fit(X1_train_full, y_train_full)
y_pred_test_1_ridge = ridge_model.predict(X1_test_full)
y_pred_full_1_ridge = ridge_model.predict(X1_full)

# Train and predict with Group 2 features
ridge_model.fit(X2_train_full, y_train_full)
y_pred_test_2_ridge = ridge_model.predict(X2_test_full)
y_pred_full_2_ridge = ridge_model.predict(X2_full)

# Calculate advanced metrics for Group 1
rmspe_test_1_ridge, mape_test_1_ridge, nrmse_test_1_ridge, q2_test_1_ridge, od_test_1_ridge = calculate_advanced_errors(y_test_full, y_pred_test_1_ridge)
rmspe_full_1_ridge, mape_full_1_ridge, nrmse_full_1_ridge, q2_full_1_ridge, od_full_1_ridge = calculate_advanced_errors(y_full, y_pred_full_1_ridge)

# Calculate advanced metrics for Group 2
rmspe_test_2_ridge, mape_test_2_ridge, nrmse_test_2_ridge, q2_test_2_ridge, od_test_2_ridge = calculate_advanced_errors(y_test_full, y_pred_test_2_ridge)
rmspe_full_2_ridge, mape_full_2_ridge, nrmse_full_2_ridge, q2_full_2_ridge, od_full_2_ridge = calculate_advanced_errors(y_full, y_pred_full_2_ridge)


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


for slice_name in slice_sheet_name :

    new_path_data = f'../output/predict/slice_data_{slice_name}.csv'
    fdf = pd.read_csv(new_path_data)


    depend_item = depend_items[slice_name]
    slice_parameter_name = depend_item["key"]

    pdData = depend_item['df']
    prData = pdData[features_group_2]
    print(prData)
    prData = scaler.fit_transform(prData)


    predictData = ridge_model.predict(prData)
    print(predictData)

    fdf[f'501 AI Data'] = predictData
    fdf.to_csv(new_path_data, index=False)
