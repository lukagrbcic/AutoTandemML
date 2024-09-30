import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

np.random.seed(4)

probes = np.loadtxt('bc_problem_data/initial_data/probes.txt')#[:8000, :]
bcs = np.loadtxt('bc_problem_data/initial_data/bcs.txt')#[:8000, :]

# for i in range(len(mpt)):
#     mpt[i][1] = int(mpt[i][1])


X = bcs
y = probes

# y = np.load('mpt_cp_dataset_reynolds.npy')#[:1000, :]
# X = np.load('cp_reynolds.npy')#[:1000, :]

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
    # 'eta': 0.1,
    # 'max_depth': 5,
    # 'subsample': 0.8,
    # 'colsample_bytree': 0.8,
    # 'reg_lamdba': 2
}

# Train the model
num_rounds = 2000
model = xgb.train(
    params,
    dtrain,
    num_rounds,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=10,
    verbose_eval=1
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



for i in range(20):
    
    
    plt.figure()
    plt.plot(np.arange(0, len(y_test_pred[i]), 1), y_test[i], 'r-')
    plt.plot(np.arange(0, len(y_test_pred[i]), 1), y_test_pred[i], 'g-')
    
    
    # plt.figure()
    # plt.title('mpt')
    # plt.plot(np.arange(0, len(y_test_pred[i,2:]), 1), y_test[i, 2:], 'ro')
    # plt.plot(np.arange(0, len(y_test_pred[i,2:]), 1), y_test_pred[i, 2:], 'go')
    
    # plt.figure()
    # plt.title('Re')
    # plt.plot(y_test[i, 0], 'ro')
    # plt.plot(y_test_pred[i, 0], 'go')
    # # plt.ylim(0, 1)
    
    # plt.figure()
    # plt.title('AoA')
    # plt.plot(y_test[i, 1], 'ro')
    # plt.plot(y_test_pred[i, 1], 'go')
    
    
    
    # plt.figure()
    # plt.title('mpt')
    # plt.plot(np.arange(0, len(y_test_pred[i,1:]), 1), y_test[i, 1:], 'ro')
    # plt.plot(np.arange(0, len(y_test_pred[i,1:]), 1), y_test_pred[i, 1:], 'go')
    
    # plt.figure()
    # plt.title('Re')
    # plt.plot(y_test[i, 0], 'ro')
    # plt.plot(y_test_pred[i, 0], 'go')
  


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


