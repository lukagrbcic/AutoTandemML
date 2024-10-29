import sys
import numpy as np
import joblib

sys.path.insert(0, 'src')
sys.path.insert(1, '../InverseBench/src/')


from benchmarks import *
from sklearn.metrics import mean_squared_error
from optimize_inverse import get_hyperparameters
from DNNRegressor import TorchDNNRegressor



param_dist = {
    'model_type': ['mlp'],
    'hidden_layers': [[64], [128], [256], [128, 128],
                      [256, 256], [512, 512], [64, 128, 64],
                      [128, 256, 128], [256, 512, 256], [64, 128, 256, 128, 64]],
    'dropout': [0.0, 0.2],
    'batch_norm': [False, True],
    'activation': ['relu', 'leaky_relu'],
    'epochs': [100, 200, 300, 1000],
    'batch_size': [32, 64],
    'learning_rate': [0.001, 0.01, 0.1],
    'input_scaler': [None, 'MinMax', 'Standard'],
    'output_scaler': [None, 'MinMax', 'Standard'],
    'output_activation': [None]
}

bench = 'airfoils'
test_input = np.load(f'../InverseBench/test_data/{bench}_data/input_test_data.npy')
test_output = np.load(f'../InverseBench/test_data/{bench}_data/output_test_data.npy')


forward_model = 'forward_model.pkl'
# print (forward_model.estimators_)
# # print(forward_model)
# sys.exit()
x_sampled = np.load('x_hf.npy')
y_sampled = np.load('y_hf.npy')

# print (forward_model.predict(x_sampled))

# hyperparemeters = get_hyperparameters(x_sampled, y_sampled, 
#                                 param_dist, n_iter=5).run()

hyperparemeters = get_hyperparameters(y_sampled, x_sampled, 
                                param_dist, n_iter=2, 
                                forward_model=forward_model).run()


      
print ('Training model')
model = TorchDNNRegressor(input_size=np.shape(y_sampled)[1],
                          output_size=np.shape(x_sampled)[1], 
                          hidden_layers=hyperparemeters['hidden_layers'],
                          output_activation=hyperparemeters['output_activation'],
                          output_scaler=hyperparemeters['output_scaler'],
                          model_type=hyperparemeters['model_type'],
                          learning_rate=hyperparemeters['learning_rate'],
                          input_scaler=hyperparemeters['input_scaler'],
                          epochs=hyperparemeters['epochs'],
                          dropout=hyperparemeters['dropout'],
                          batch_size=hyperparemeters['batch_size'],
                          batch_norm=hyperparemeters['batch_norm'],
                          activation=hyperparemeters['activation'],
                          forward_model=forward_model)

                          
model.fit(y_sampled, x_sampled)


preds = model.predict(test_input)

name = 'airfoil_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)
lb, ub = f.get_bounds()
def evaluation_function(x):
    
    value = f.evaluate(x)
    
    return value

predictions = evaluation_function(preds)


rmse_list = np.array([np.sqrt(mean_squared_error(test_output[i], predictions[i])) for i in range(len(preds))])
                          
print ('MEAN', np.mean(rmse_list))         
print ('MAX', np.max(rmse_list))         

import matplotlib.pyplot as plt

for i in range(20):
    plt.plot(np.arange(0, len(predictions[i]), 1), predictions[i], 'g-')
    plt.plot(np.arange(0, len(test_input[i]), 1), test_input[i], 'g-')









