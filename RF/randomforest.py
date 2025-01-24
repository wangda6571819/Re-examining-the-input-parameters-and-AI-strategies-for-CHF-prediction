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
