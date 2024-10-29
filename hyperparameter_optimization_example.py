import sys
import numpy as np
import joblib

sys.path.insert(0, 'src')
sys.path.insert(1, '../InverseBench/src/')


from benchmarks import *
from sklearn.metrics import mean_squared_error
from optimize_inverse import get_hyperparameters
from DNNRegressor import TorchDNNRegressor
from model_factory import ModelFactory
from get_forward import forwardDNN
from get_inverse import inverseDNN
import torch



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
    'input_scaler': ['MinMax', 'Standard'],
    'output_scaler': ['MinMax', 'Standard'],
    'output_activation': [None]
}

bench = 'airfoils'
test_input = np.load(f'../InverseBench/test_data/{bench}_data/input_test_data.npy')
test_output = np.load(f'../InverseBench/test_data/{bench}_data/output_test_data.npy')

sampler = 'model_uncertainty'
# sampler = 'random'

# forward_model = joblib.load(f'forward_model_{sampler}.pkl')

x_sampled = np.load(f'x_hf_{sampler}.npy')
y_sampled = np.load(f'y_hf_{sampler}.npy')

# x_sampled = np.load(f'x_mf_{sampler}.npy')
# y_sampled = np.load(f'y_mf_{sampler}.npy')


fwd_hyperparameters = get_hyperparameters(x_sampled, y_sampled, 
                                param_dist, n_iter=200).run()

get_forward_dnn = forwardDNN(x_sampled, y_sampled, fwd_hyperparameters).train_save()



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
    'input_scaler': [fwd_hyperparameters['output_scaler']],
    'output_scaler': [fwd_hyperparameters['input_scaler']],
    'output_activation': [None]
}


hyperparameters = get_hyperparameters(y_sampled, x_sampled, 
                                param_dist, n_iter=200, 
                                forward_model_hyperparameters=fwd_hyperparameters).run()





print (hyperparameters)


get_inverse_dnn = inverseDNN(y_sampled, x_sampled, hyperparameters, forward_model_hyperparameters=fwd_hyperparameters).train_save()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


inverse_dnn = ModelFactory().create_model(model_type=hyperparameters['model_type'], 
                                          input_size=np.shape(y_sampled)[1], 
                                          output_size=np.shape(x_sampled)[1],
                                          hidden_layers=hyperparameters['hidden_layers'],
                                          dropout=hyperparameters['dropout'],
                                          output_activation=hyperparameters['output_activation'],
                                          batch_norm=hyperparameters['batch_norm'],
                                          activation=hyperparameters['activation'])

print (inverse_dnn)

inverse_dnn.load_state_dict(torch.load('inverseDNN.pth'))
inverse_dnn.eval()

input_scaler = joblib.load('input_scaler_inverse.pkl')
test_output_scaled = input_scaler.transform(test_output)

test_output_torch = torch.tensor(test_output_scaled, dtype=torch.float32).to(device)

# test_output_torch = torch.tensor(test_output, dtype=torch.float32).to(device)


preds = inverse_dnn(test_output_torch)
preds = preds.detach().cpu().numpy()
print (preds)
scaler = joblib.load('output_scaler_inverse.pkl')
preds = scaler.inverse_transform(preds)
print (preds)



name = 'airfoil_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)
lb, ub = f.get_bounds()
def evaluation_function(x):
    
    value = f.evaluate(x)
    
    return value

predictions = evaluation_function(preds)


rmse_list = np.array([np.sqrt(mean_squared_error(test_output[i], predictions[i])) for i in range(len(predictions))])
                          
print ('MEAN', np.mean(rmse_list))         
print ('MAX', np.max(rmse_list))         

# import matplotlib.pyplot as plt

# for i in range(5):
#     plt.figure()
#     plt.plot(np.arange(0, len(predictions[i]), 1), predictions[i], 'g-')
#     plt.plot(np.arange(0, len(test_output[i]), 1), test_output[i], 'r-')









