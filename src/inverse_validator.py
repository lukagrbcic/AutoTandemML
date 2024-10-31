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


class inverse_model_analysis:
    
    def __init__(self, test_input, test_output, benchmark,
                       model_config_path='inverseDNN', 
                       inverse_model_path='inverseDNN', 
                       scalers_path='inverseDNN'):
        
        self.test_input = test_input
        self.test_output = test_output
        self.benchmark = benchmark
        self.model_config_path = model_config_path
        self.inverse_model_path = inverse_model_path
        self.scalers_path = scalers_path
        
        self.model = load_model(self.benchmark).load()
    
    def evaluate(self, x):
        
        return benchmark_functions(self.benchmark, self.model).evaluate(x)
        

    def get_inverse_dnn(self):
        
        hyperparameters = np.load(f'{self.model_config_path}model_config.npy', allow_pickle=True).item()
        
        inverse_dnn = ModelFactory().create_model(model_type=hyperparameters['model_type'], 
                                                  input_size=np.shape(y_sampled)[1], 
                                                  output_size=np.shape(x_sampled)[1],
                                                  hidden_layers=hyperparameters['hidden_layers'],
                                                  dropout=hyperparameters['dropout'],
                                                  output_activation=hyperparameters['output_activation'],
                                                  batch_norm=hyperparameters['batch_norm'],
                                                  activation=hyperparameters['activation'])
        
        inverse_dnn.load_state_dict(torch.load(f'{self.inverse_model_path}/inverseDNN.pth'))
        inverse_dnn.eval()
        
        return inverse_dnn
    
    def get_scalers(self):

        input_scaler = joblib.load(f'{self.scalers_path}/input_scaler_inverse.pkl')
        output_scaler = joblib.load(f'{self.scalers_path}/output_scaler_inverse.pkl')

        return input_scaler, output_scaler
    
    



        # test_output_scaled = input_scaler.transform(test_output)
        
        # test_output_torch = torch.tensor(test_output_scaled, dtype=torch.float32).to(device)
        
        
        # preds = inverse_dnn(test_output_torch)
        # preds = preds.detach().cpu().numpy()
        # preds = scaler.inverse_transform(preds)
        
        
        # predictions = evaluation_function(preds)
        