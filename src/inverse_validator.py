import numpy as np
from sklearn.metrics import *
import sys
import joblib
import torch

sys.path.insert(0, 'src')
sys.path.insert(1, '../InverseBench/src/')

from benchmarks import *
from model_factory import ModelFactory


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


name = 'airfoil_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)

# name = 'inconel_benchmark'
# model = load_model(name).load()
# f = benchmark_functions(name, model)

# name = 'friedman_multioutput_benchmark'
# f = benchmark_functions(name)

def evaluation_function(x):
    value = f.evaluate(x)
    return value





class inverse_model_analysis:
    
    def __init__(self, test_input, test_output, benchmark,
                       model_config_path='inverseDNN/', 
                       inverse_path='inverseDNN/', 
                       scaler_path='inverseDNN/'):
        
        self.test_input = test_input
        self.test_output = test_output
        self.benchmark = benchmark
        self.model_config_path = model_config_path
        self.inverse_path = inverse_path
        self.scalers_path = scalers_path
        
        if self.benchmark is not 'friedman_multioutput_benchmark':
            self.model = load_model(self.benchmark).load()
    
    def model_eval(self, x):
        
        f = benchmark_functions(self.benchmark)
        
        return f.evaluate(x)
    
    def func_eval(self, x)

        
        
        
        
        
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
        