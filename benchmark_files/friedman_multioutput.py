import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

np.random.seed(4)

def friedman_multioutput(n_samples, n_targets, noise_level=0.1):
    """
    Generate a one-to-many multioutput version of the Friedman #1 function.
    
    Parameters:
    - n_samples: int, number of samples to generate
    - n_targets: int, number of target variables
    - noise_level: float, standard deviation of the noise
    
    Returns:
    - X: array of shape (n_samples, 5), input features
    - y: array of shape (n_samples, n_targets), target variables
    """
    
    # Generate one input feature
    X = np.random.uniform(0, 1, (n_samples, 5))
    
    # Initialize output array
    y = np.zeros((n_samples, n_targets))
    
    for i in range(n_targets):
        # Modify coefficients for each target
        # a = np.random.uniform(5, 10)  # Instead of (0, 10)
        # b = np.random.uniform(25, 35)  # Instead of (20, 40)
        # c = np.random.uniform(0.4, 0.6)  # Instead of (0, 1)
        # d = np.random.uniform(2, 3)  # Instead of (1, 4)
        # e = np.random.uniform(0.4, 0.6)  # Instead of (0, 1)
        
        a = np.random.uniform(5, 8)  # Instead of (0, 10)
        b = np.random.uniform(25, 28)  # Instead of (20, 40)
        c = np.random.uniform(0.4, 0.5)  # Instead of (0, 1)
        d = np.random.uniform(2, 3)  # Instead of (1, 4)
        e = np.random.uniform(0.4, 0.5)  # Instead of (0, 1)

        
        # Calculate target using modified Friedman #1 function
        y[:, i] = (a * np.sin(np.pi * X[:, 0] * X[:, 1]) +
                   b * (X[:, 2] - 0.5)**2 +
                   c * X[:, 3] +
                   d * X[:, 4] +
                   e * np.mean(X, axis=1) +  # Additional term using mean of all features
                   np.random.normal(0, noise_level, n_samples))
    
    return X, y

# Usage example
n_samples = 100000
n_targets = 10
y, X = friedman_multioutput(n_samples, n_targets)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=23)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, random_state=23)

# X_train, X_test, y_train, y_test = train_test_split(y, X, test_size=0.3, shuffle=True, random_state=23)
# X_train_, X_val, y_train_, y_val = train_test_split(y_train, X_train, test_size=0.1, shuffle=True, random_state=23)



# Create DMatrix objects for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)
# dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lamdba': 10
}

# Train the model
num_rounds = 2000
model = xgb.train(
    params,
    dtrain,
    num_rounds,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=10,
    verbose_eval=10
)


def calculate_rmse(y_true, y_pred):   
    rmse = np.mean(np.array([np.sqrt(mean_squared_error(y_true[i], y_pred[i])) for i in range(len(y_pred))]))
    return rmse


def calculate_mape(y_true, y_pred):   
    mape = np.mean(np.array([mean_absolute_percentage_error(y_true[i], y_pred[i]) for i in range(len(y_pred))]))
    return mape



# Evaluate on validation set
y_val_pred = model.predict(dval)
val_rmse_per_target = calculate_rmse(y_val, y_val_pred)
print("Validation RMSE per target:", val_rmse_per_target)
print("Average Validation RMSE:", np.mean(val_rmse_per_target))

# Evaluate on test set
y_test_pred = model.predict(dtest)
test_rmse_per_target = calculate_rmse(y_test, y_test_pred)
print("\nTest RMSE per target:", test_rmse_per_target)
print("Average Test RMSE:", np.mean(test_rmse_per_target))


# Evaluate on test set
# y_test_pred = model.predict(dtest)
test_mape_per_target = calculate_mape(y_test, y_test_pred)
print("\nTest MAPE per target:", test_mape_per_target)
print("Average Test MAPE:", np.mean(test_mape_per_target))



for i in range(5):
    plt.figure()
    plt.plot(np.arange(0, len(y_test_pred[i]), 1), y_test[i], 'r-')
    plt.plot(np.arange(0, len(y_test_pred[i]), 1), y_test_pred[i], 'g-')
    # plt.ylim(0, 1)



# val_scores = model.eval_set([(dval, 'val')])

# # # Extract RMSE values and convert to a list
# val_rmse = [float(score.split(':')[1]) for score in val_scores]

# # Plot the validation loss
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(val_rmse) + 1), val_rmse)
# plt.title('XGBoost Validation RMSE over Iterations')
# plt.xlabel('Iterations')
# plt.ylabel('Validation RMSE')
# plt.show()


