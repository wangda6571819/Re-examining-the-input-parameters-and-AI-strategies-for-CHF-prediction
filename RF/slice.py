import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data

file_path = '../output/AIData_ALL_model_Transformer_110.csv'
data = pd.read_csv(file_path)

# Convert relevant columns to numeric and handle missing values

data['CHF'] = pd.to_numeric(data['CHF'], errors='coerce')
data['Tube Diameter'] = pd.to_numeric(data['Tube Diameter'], errors='coerce')
data['Heated Length'] = pd.to_numeric(data['Heated Length'], errors='coerce')
data['Pressure'] = pd.to_numeric(data['Pressure'], errors='coerce')
data['Mass Flux'] = pd.to_numeric(data['Mass Flux'], errors='coerce')
data['Inlet Subcooling'] = pd.to_numeric(data['Inlet Subcooling'], errors='coerce')

data_clean = data.dropna(subset=['CHF', 'Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Inlet Subcooling'])

# Define features and target variable

feature_columns = ['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Inlet Subcooling']

X = data_clean[feature_columns]
y = data_clean['CHF']

# Standardize the features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the middle sampling function

def middle_sampling(group, total_frac=0.2, middle_frac=0.8):
    # Define the number of samples to pick as a fraction of the total group size

    total_samples = max(int(round(len(group) * total_frac)), 1)  # Ensure at least 1 sample per group


    # Calculate the start and end indices of the middle portion to avoid extremes if possible

    start_index = int(round(len(group) * (1 - middle_frac) / 2))
    end_index = len(group) - start_index


    # If group is too small and ends overlap, adjust to use the full group, else use middle

    if start_index >= end_index:
        sampled_indices = range(0, len(group))
    else:
        # Calculate the sample interval within the middle portion

        sample_interval = (end_index - start_index) / total_samples

        sampled_indices = [int(round(start_index + i * sample_interval)) for i in range(total_samples)]

    # Ensure no out of bounds or duplicate indices for small groups

    sampled_indices = list(sorted(set(sampled_indices)))
    sampled_indices = [idx if idx < len(group) else len(group) - 1 for idx in sampled_indices]

    return group.iloc[sampled_indices]

# Apply middle sampling to the dataset

data_sampled = data_clean.groupby('CHF', group_keys=False).apply(middle_sampling)

# Define features and target variable for the sampled data


X_sampled = data_sampled[feature_columns]
y_sampled = data_sampled['CHF']

# Standardize the features for the sampled data

X_sampled_scaled = scaler.fit_transform(X_sampled)

# Split the sampled data into training (64%), validation (16%), and test (20%) sets

X_train, X_test, y_train, y_test = train_test_split(X_sampled_scaled, y_sampled, test_size=0.20, random_state=42)

# Train the Random Forest model on the training set

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set

y_test_pred = rf_model.predict(X_test)

# Calculate performance metrics for the test set

rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
nrmse_test = rmse_test / (y_test.max() - y_test.min())
rmspe_test = np.sqrt(np.mean(np.square((y_test - y_test_pred) / y_test))) * 100

mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

ss_res_test = np.sum(np.square(y_test - y_test_pred))
ss_tot_test = np.sum(np.square(y_test - np.mean(y_test)))
q2_test = 1 - (ss_res_test / ss_tot_test)

# Evaluate on the entire dataset

y_pred_rf_whole = rf_model.predict(X_scaled)

rmse_whole = mean_squared_error(y, y_pred_rf_whole, squared=False)
nrmse_whole = rmse_whole / (y.max() - y.min())
rmspe_whole = np.sqrt(np.mean(np.square((y - y_pred_rf_whole) / y))) * 100

mape_whole = np.mean(np.abs((y - y_pred_rf_whole) / y)) * 100

ss_res_whole = np.sum(np.square(y - y_pred_rf_whole))
ss_tot_whole = np.sum(np.square(y - np.mean(y)))
q2_whole = 1 - (ss_res_whole / ss_tot_whole)

def output_indicator(pre, real):
    mean = np.average(pre / real)
    std = np.std(pre / real)
    rmspe = np.sqrt(np.mean(np.square((pre - real) / real)))
    mape = np.mean(np.abs((pre - real) / real))
    print("Mean P/M:", mean)
    print("Std P/M:", std)
    print("RMSPE:", rmspe)
    print("MAPE:", mape)
    rmse = np.sqrt(np.mean(np.square(real - pre)))
    nrmse_mean = rmse / np.mean(real)
    print("NRMSE:", nrmse_mean)
    mu = np.mean(real)
    numerator = np.sum((real - pre) ** 2)
    denominator = np.sum((real - mu) ** 2)
    EQ2 = numerator / denominator

    print("Q2:", EQ2)
    result = [rmspe, mape, nrmse_mean, EQ2]
    return [float(element) for element in result]

print(np.mean(output_indicator(y_pred_rf_whole, y)))

# Output the results

print(rmse_test, nrmse_test, rmspe_test, mape_test, q2_test, rmse_whole, nrmse_whole, rmspe_whole, mape_whole, q2_whole)



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


    plt.figure(figsize=(5,3),dpi=200)

    prData = pdData[feature_columns]
    print(prData)
    prData_scaled = scaler.fit_transform(prData)
    predictData = rf_model.predict(prData_scaled)
    print(predictData)

    # pdData[f'401 AI Data'] = predictData

    # pdData.to_csv(f'./slice_data_{slice_name}.csv', index=False)
    fdf[f'401 AI Data'] = predictData
    fdf.to_csv(new_path_data, index=False)
