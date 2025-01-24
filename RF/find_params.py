# Re-import necessary libraries and reset the environment
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load the data again
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
X = data_clean[['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Inlet Subcooling']]
y = data_clean['CHF']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Create the RandomizedSearchCV object
rf_random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=50,  # Number of iterations to perform
    cv=5,  # 5-fold cross-validation
    verbose=2,  # Print progress
    n_jobs=-1,  # Use all available cores
    random_state=42
)

# Fit the randomized search model
rf_random_search.fit(X_train, y_train)

# Get the best parameters from the randomized search
best_params_rf = rf_random_search.best_params_

# Train the final model with the best parameters
rf_tuned_model = RandomForestRegressor(**best_params_rf, random_state=42)
rf_tuned_model.fit(X_train, y_train)

# Predict on the test set using the tuned Random Forest model
y_test_pred_rf_tuned = rf_tuned_model.predict(X_test)

# Calculate RMSE and NRMSE for the test set
rmse_test_rf_tuned = mean_squared_error(y_test, y_test_pred_rf_tuned, squared=False)
nrmse_test_rf_tuned = rmse_test_rf_tuned / y_test.mean()

# Calculate RMSPE and MAPE for the test set
rmspe_test_rf_tuned = np.sqrt(np.mean(np.square((y_test - y_test_pred_rf_tuned) / y_test))) * 100
mape_test_rf_tuned = np.mean(np.abs((y_test - y_test_pred_rf_tuned) / y_test)) * 100

# Calculate Q² (Coefficient of Determination) for the test set
ss_res_test_rf_tuned = np.sum(np.square(y_test - y_test_pred_rf_tuned))
ss_tot_test_rf_tuned = np.sum(np.square(y_test - np.mean(y_test)))
q2_test_rf_tuned = 1 - (ss_res_test_rf_tuned / ss_tot_test_rf_tuned)

# Output the performance metrics and the best parameters
print(best_params_rf, rmse_test_rf_tuned, nrmse_test_rf_tuned, rmspe_test_rf_tuned, mape_test_rf_tuned, q2_test_rf_tuned)
