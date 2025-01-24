import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

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

# Initialize Random Forest with default configurations
rf_model = RandomForestRegressor(random_state=42)

# Train and predict with Group 1 features
rf_model.fit(X1_train_full, y_train_full)
y_pred_test_1_rf = rf_model.predict(X1_test_full)
y_pred_full_1_rf = rf_model.predict(X1_full)

# Train and predict with Group 2 features
rf_model.fit(X2_train_full, y_train_full)
y_pred_test_2_rf = rf_model.predict(X2_test_full)
y_pred_full_2_rf = rf_model.predict(X2_full)

# Calculate advanced metrics for Group 1
rmspe_test_1_rf, mape_test_1_rf, nrmse_test_1_rf, q2_test_1_rf, od_test_1_rf = calculate_advanced_errors(y_test_full, y_pred_test_1_rf)
rmspe_full_1_rf, mape_full_1_rf, nrmse_full_1_rf, q2_full_1_rf, od_full_1_rf = calculate_advanced_errors(y_full, y_pred_full_1_rf)

# Calculate advanced metrics for Group 2
rmspe_test_2_rf, mape_test_2_rf, nrmse_test_2_rf, q2_test_2_rf, od_test_2_rf = calculate_advanced_errors(y_test_full, y_pred_test_2_rf)
rmspe_full_2_rf, mape_full_2_rf, nrmse_full_2_rf, q2_full_2_rf, od_full_2_rf = calculate_advanced_errors(y_full, y_pred_full_2_rf)

# Store the results for Random Forest
rf_results = {
    'Group 1': {'RMSPE Test': rmspe_test_1_rf, 'MAPE Test': mape_test_1_rf, 'NRMSE Test': nrmse_test_1_rf, '1-Q² Test': q2_test_1_rf, 'OD Test': od_test_1_rf,
                'RMSPE Full': rmspe_full_1_rf, 'MAPE Full': mape_full_1_rf, 'NRMSE Full': nrmse_full_1_rf, '1-Q² Full': q2_full_1_rf, 'OD Full': od_full_1_rf},
    'Group 2': {'RMSPE Test': rmspe_test_2_rf, 'MAPE Test': mape_test_2_rf, 'NRMSE Test': nrmse_test_2_rf, '1-Q² Test': q2_test_2_rf, 'OD Test': od_test_2_rf,
                'RMSPE Full': rmspe_full_2_rf, 'MAPE Full': mape_full_2_rf, 'NRMSE Full': nrmse_full_2_rf, '1-Q² Full': q2_full_2_rf, 'OD Full': od_full_2_rf}
}

# Output all predicted values and ensure they are stored separately for test and full datasets
predicted_values_test = {
    'Group 1 Test Predictions': y_pred_test_1_rf,
    'Group 2 Test Predictions': y_pred_test_2_rf
}

predicted_values_full = {
    'Group 1 Full Predictions': y_pred_full_1_rf,
    'Group 2 Full Predictions': y_pred_full_2_rf
}

# Print the results
print("Random Forest Results:")
print(rf_results)

print("\nTest Set Predicted Values:")
for key, value in predicted_values_test.items():
    print(f"{key}: {value}")

print("\nFull Dataset Predicted Values:")
for key, value in predicted_values_full.items():
    print(f"{key}: {value}")


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

for slice_name in slice_sheet_name :

    new_path_data = f'../output/predict/slice_data_{slice_name}.csv'
    fdf = pd.read_csv(new_path_data)


    depend_item = depend_items[slice_name]
    slice_parameter_name = depend_item["key"]

    pdData = depend_item['df']


    prData = pdData[features_group_2]
    print(prData)
    # prData_scaled = scaler.fit_transform(prData)
    predictData = rf_model.predict(prData)
    print(predictData)

    # pdData[f'401 AI Data'] = predictData

    # pdData.to_csv(f'./slice_data_{slice_name}.csv', index=False)
    fdf[f'401 AI Data'] = predictData
    fdf.to_csv(new_path_data, index=False)
