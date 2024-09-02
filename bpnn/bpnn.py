import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the data
file_path = 'AIData_ALL_model_Transformer_110.csv'  # Replace with the actual path to your CSV file
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

# Initialize BPNN with default configurations
bpnn_model = MLPRegressor(random_state=42, max_iter=1000)

# Train and predict with Group 1 features
bpnn_model.fit(X1_train_full, y_train_full)
y_pred_test_1_bpnn = bpnn_model.predict(X1_test_full)
y_pred_full_1_bpnn = bpnn_model.predict(X1_full)

# Calculate advanced metrics for Group 1
rmspe_test_1_bpnn, mape_test_1_bpnn, nrmse_test_1_bpnn, q2_test_1_bpnn, od_test_1_bpnn = calculate_advanced_errors(y_test_full, y_pred_test_1_bpnn)
rmspe_full_1_bpnn, mape_full_1_bpnn, nrmse_full_1_bpnn, q2_full_1_bpnn, od_full_1_bpnn = calculate_advanced_errors(y_full, y_pred_full_1_bpnn)

# Train and predict with Group 2 features
bpnn_model.fit(X2_train_full, y_train_full)
y_pred_test_2_bpnn = bpnn_model.predict(X2_test_full)
y_pred_full_2_bpnn = bpnn_model.predict(X2_full)

# Calculate advanced metrics for Group 2
rmspe_test_2_bpnn, mape_test_2_bpnn, nrmse_test_2_bpnn, q2_test_2_bpnn, od_test_2_bpnn = calculate_advanced_errors(y_test_full, y_pred_test_2_bpnn)
rmspe_full_2_bpnn, mape_full_2_bpnn, nrmse_full_2_bpnn, q2_full_2_bpnn, od_full_2_bpnn = calculate_advanced_errors(y_full, y_pred_full_2_bpnn)

# Store the results for BPNN
bpnn_results = {
    'Group 1': {'RMSPE Test': rmspe_test_1_bpnn, 'MAPE Test': mape_test_1_bpnn, 'NRMSE Test': nrmse_test_1_bpnn, '1-Q² Test': q2_test_1_bpnn, 'OD Test': od_test_1_bpnn,
                'RMSPE Full': rmspe_full_1_bpnn, 'MAPE Full': mape_full_1_bpnn, 'NRMSE Full': nrmse_full_1_bpnn, '1-Q² Full': q2_full_1_bpnn, 'OD Full': od_full_1_bpnn},
    'Group 2': {'RMSPE Test': rmspe_test_2_bpnn, 'MAPE Test': mape_test_2_bpnn, 'NRMSE Test': nrmse_test_2_bpnn, '1-Q² Test': q2_test_2_bpnn, 'OD Test': od_test_2_bpnn,
                'RMSPE Full': rmspe_full_2_bpnn, 'MAPE Full': mape_full_2_bpnn, 'NRMSE Full': nrmse_full_2_bpnn, '1-Q² Full': q2_full_2_bpnn, 'OD Full': od_full_2_bpnn}
}

print(bpnn_results)
