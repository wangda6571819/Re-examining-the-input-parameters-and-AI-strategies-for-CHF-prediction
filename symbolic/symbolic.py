from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from numba import jit

# Load your data
data = pd.read_csv('../data/all_properties_data.csv')  # Replace with your file path

# Define the input features (including Inlet Subcooling)
X = data[['Tube Diameter', 'Heated Length', 'Pressure', 'Mass Flux', 'Inlet Subcooling']]
y = data['CHF']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test = X
y_test = y

# mae
@jit(nopython=True)
def _mape(y, y_pred, w):
    """Calculate the mean absolute percentage error."""
    diffs = np.abs(np.divide((np.maximum(0.001, y) - np.maximum(0.001, y_pred)),
                             np.maximum(0.001, y)))
    return 100. * np.average(diffs, weights=w)
# rmse
@jit(nopython=True)
def _rmse(y, y_pred, w):
    """Calculate the root mean squared error."""
    diffs = np.square(y - y_pred)
    return np.sqrt(np.average(diffs, weights=w))

@jit(nopython=True)
def _mse(y, y_pred, w):
    """Calculate the mean squared error."""
    diffs = np.square(y - y_pred)
    return np.average(diffs, weights=w)
metric = make_fitness(function=_mse, greater_is_better=False)

# Define a custom exponential function with overflow protection using JIT
@jit(nopython=True)
def custom_exp(x):
    return np.where(x > 0, np.exp(np.minimum(x, 50)), 0)  # Cap x to prevent overflow
# Create a custom function for exponential
exp_function = make_function(function=custom_exp, name='exp', arity=1)


# function_set = ('add','sub','mul','div','sqrt','log','abs','neg','inv','max','min','sin','cos','tan',exp_function)
function_set = ('add', 'sub', 'mul', 'div', 'sqrt', 'log',exp_function)
# Initialize the Symbolic Regressor
symbolic_model = SymbolicRegressor(
    population_size=2000,             
    generations=50,                   
    stopping_criteria=0.001,          
    function_set=function_set, 
    p_crossover=0.7,                  # Reduced to allow room for mutations
    p_subtree_mutation=0.1,           
    p_hoist_mutation=0.05,            
    p_point_mutation=0.1,             
    max_samples=0.9,                  
    verbose=1,                        
    random_state=42,
)

# Fit the symbolic regression model
symbolic_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_symbolic = symbolic_model.predict(X_test)

# Evaluate the performance
mse_symbolic = mean_squared_error(y_test, y_pred_symbolic)
r2_symbolic = r2_score(y_test, y_pred_symbolic)

# Print the results
print(f"Mean Squared Error: {mse_symbolic}")
print(f"R-squared: {r2_symbolic}")

print('Discovered best Symbolic Expression\t', symbolic_model._program)
print('other ：')
for program in symbolic_model._programs[-1][:20]:
    print(f'program.raw_fitness : {program.raw_fitness_}  program : {program} ')


# def calculate_advanced_errors(y_true, y_pred):
#     rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
#     mape = np.mean(np.abs((y_true - y_pred) / y_true))
#     nrmse = np.sqrt(mean_squared_error(y_true, y_pred)) / y_true.mean()
#     q2 = 1 - r2_score(y_true, y_pred)
#     od = np.mean([rmspe, mape, nrmse, q2])
#     return rmspe, mape, nrmse, q2, od

# rmspe_full_1_ridge, mape_full_1_ridge, nrmse_full_1_ridge, q2_full_1_ridge, od_full_1_ridge = calculate_advanced_errors(X_test, y_pred_symbolic)

# print(rmspe_full_1_ridge, mape_full_1_ridge, nrmse_full_1_ridge, q2_full_1_ridge, od_full_1_ridge)