import sys
import numpy as np
import joblib

sys.path.insert(0, 'src')
sys.path.insert(1, '../InverseBench/src/')


from benchmarks import *
from sklearn.metrics import *
from optimize_inverse import get_hyperparameters
from DNNRegressor import TorchDNNRegressor
from model_factory import ModelFactory
from get_forward import forwardDNN
from get_inverse import inverseDNN
import torch

rmse_rand = []
nmax_ae_rand = []
mape_rand = []
r2_rand = []


rmse_unc = []
nmax_ae_unc = []
mape_unc = []
r2_unc = []

combinations=100

    
name = 'airfoil_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)

# name = 'inconel_benchmark'
# model = load_model(name).load()
# f = benchmark_functions(name, model)

# name = 'friedman_multioutput_benchmark'
# f = benchmark_functions(name)


lb, ub = f.get_bounds()

def evaluation_function(x):
    value = f.evaluate(x)
    return value

x_sampled_rand = np.random.uniform(lb, ub, size=(200, len(lb)))
y_sampled_rand = evaluation_function(x_sampled_rand)
    
bench = 'airfoils'
test_input = np.load(f'../InverseBench/test_data/{bench}_data/input_test_data.npy')
test_output = np.load(f'../InverseBench/test_data/{bench}_data/output_test_data.npy')
    


for i in range(15):
    print ('Run', i)

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
    

    
    x_sampled = x_sampled_rand
    y_sampled = y_sampled_rand
    
    
    
    fwd_hyperparameters = get_hyperparameters(x_sampled, y_sampled, 
                                    param_dist, n_iter=combinations).run()
    
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
                                    param_dist, n_iter=combinations, 
                                    forward_model_hyperparameters=fwd_hyperparameters).run()
    
    
    
    
    
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
    
    inverse_dnn.load_state_dict(torch.load('inverseDNN.pth'))
    inverse_dnn.eval()
    
    input_scaler = joblib.load('input_scaler_inverse.pkl')
    test_output_scaled = input_scaler.transform(test_output)
    
    test_output_torch = torch.tensor(test_output_scaled, dtype=torch.float32).to(device)
    
    
    preds = inverse_dnn(test_output_torch)
    preds = preds.detach().cpu().numpy()
    scaler = joblib.load('output_scaler_inverse.pkl')
    preds = scaler.inverse_transform(preds)
    
    
    predictions = evaluation_function(preds)
    
    
    def nmax_ae(y_true, y_pred):
        
        nmaxae = np.max(np.abs(y_true - y_pred), axis=0)/np.max(np.abs(y_true - np.mean(y_true, axis=0)))
        
        return np.mean(nmaxae)      
    
    rmse_rand.append(np.sqrt(mean_squared_error(test_output, predictions)))      
    nmax_ae_rand.append(nmax_ae(test_output, predictions))  
    mape_rand.append(mean_absolute_percentage_error(test_output, predictions))        
    r2_rand.append(r2_score(test_output, predictions))    

    
    #unc
    
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
    

    
    sampler = 'model_uncertainty'
    
    x_sampled = np.load(f'x_hf_{sampler}_{bench}.npy')
    y_sampled = np.load(f'y_hf_{sampler}_{bench}.npy')
        
    
    fwd_hyperparameters = get_hyperparameters(x_sampled, y_sampled, 
                                    param_dist, n_iter=combinations).run()
    
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
                                    param_dist, n_iter=combinations, 
                                    forward_model_hyperparameters=fwd_hyperparameters).run()
    
    
    
    
    
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
    
    inverse_dnn.load_state_dict(torch.load('inverseDNN.pth'))
    inverse_dnn.eval()
    
    input_scaler = joblib.load('input_scaler_inverse.pkl')
    test_output_scaled = input_scaler.transform(test_output)
    
    test_output_torch = torch.tensor(test_output_scaled, dtype=torch.float32).to(device)
    
    
    preds = inverse_dnn(test_output_torch)
    preds = preds.detach().cpu().numpy()
    scaler = joblib.load('output_scaler_inverse.pkl')
    preds = scaler.inverse_transform(preds)
    
    
    predictions = evaluation_function(preds)
    
    
    # rmse_list = np.array([np.sqrt(mean_squared_error(test_output[i], predictions[i])) for i in range(len(predictions))])
                              
    # print ('MEAN', np.mean(rmse_list))         
    # print ('MAX', np.max(rmse_list))   
    
    def nmax_ae(y_true, y_pred):
        
        nmaxae = np.max(np.abs(y_true - y_pred), axis=0)/np.max(np.abs(y_true - np.mean(y_true, axis=0)))
        
        return np.mean(nmaxae)      
    
    rmse_unc.append(np.sqrt(mean_squared_error(test_output, predictions)))      
    nmax_ae_unc.append(nmax_ae(test_output, predictions))  
    mape_unc.append(mean_absolute_percentage_error(test_output, predictions))        
    r2_unc.append(r2_score(test_output, predictions)) 


print ('rmse rand', np.mean(rmse_rand))        
print ('nmax_ae rand', np.mean(nmax_ae_rand))      
print ('mape rand', np.mean(mape_rand))         
print ('r2 rand', np.mean(r2_rand))      


print ('rmse unc', np.mean(rmse_unc))        
print ('nmax_ae unc', np.mean(nmax_ae_unc))      
print ('mape unc', np.mean(mape_unc))         
print ('r2 unc', np.mean(r2_unc))  







