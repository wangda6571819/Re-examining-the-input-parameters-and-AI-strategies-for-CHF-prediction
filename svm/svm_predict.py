import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
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

# Initialize ε-SVM with a linear kernel for simplicity
svm_model_simplified = SVR(kernel='linear')

# Train and predict with Group 1 features
svm_model_simplified.fit(X1_train_full, y_train_full)
y_pred_test_1_svm_simplified = svm_model_simplified.predict(X1_test_full)
y_pred_full_1_svm_simplified = svm_model_simplified.predict(X1_full)

# Calculate advanced metrics for Group 1
rmspe_test_1_svm_simplified, mape_test_1_svm_simplified, nrmse_test_1_svm_simplified, q2_test_1_svm_simplified, od_test_1_svm_simplified = calculate_advanced_errors(y_test_full, y_pred_test_1_svm_simplified)
rmspe_full_1_svm_simplified, mape_full_1_svm_simplified, nrmse_full_1_svm_simplified, q2_full_1_svm_simplified, od_full_1_svm_simplified = calculate_advanced_errors(y_full, y_pred_full_1_svm_simplified)

# Train and predict with Group 2 features
svm_model_simplified.fit(X2_train_full, y_train_full)
y_pred_test_2_svm_simplified = svm_model_simplified.predict(X2_test_full)
y_pred_full_2_svm_simplified = svm_model_simplified.predict(X2_full)

# Calculate advanced metrics for Group 2
rmspe_test_2_svm_simplified, mape_test_2_svm_simplified, nrmse_test_2_svm_simplified, q2_test_2_svm_simplified, od_test_2_svm_simplified = calculate_advanced_errors(y_test_full, y_pred_test_2_svm_simplified)
rmspe_full_2_svm_simplified, mape_full_2_svm_simplified, nrmse_full_2_svm_simplified, q2_full_2_svm_simplified, od_full_2_svm_simplified = calculate_advanced_errors(y_full, y_pred_full_2_svm_simplified)


