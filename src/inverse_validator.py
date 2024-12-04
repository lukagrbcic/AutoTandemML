import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
import sys
import joblib
import torch

sys.path.insert(0, 'src')

from InverseBench.benchmarks import *
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
        
    def load_inverse_dnn(self):
        
        hyperparameters = np.load(f'{self.model_config_path}/model_config.npy', allow_pickle=True).item()
        
        inverse_dnn = ModelFactory().create_model(model_type=hyperparameters['model_type'], 
                                                  input_size=np.shape(self.test_output)[1], 
                                                  output_size=np.shape(self.test_input)[1],
                                                  hidden_layers=hyperparameters['hidden_layers'],
                                                  dropout=hyperparameters['dropout'],
                                                  output_activation=hyperparameters['output_activation'],
                                                  batch_norm=hyperparameters['batch_norm'],
                                                  activation=hyperparameters['activation'])
        
        inverse_dnn.load_state_dict(torch.load(f'{self.inverse_model_path}/inverseDNN.pth'))
        inverse_dnn.eval()
        
        return inverse_dnn
    

    def load_forward_dnn(self):
        
        hyperparameters = np.load(f'forward_DNN/model_config.npy', allow_pickle=True).item()
        
        forward_dnn = ModelFactory().create_model(model_type=hyperparameters['model_type'], 
                                                  input_size=np.shape(self.test_input)[1], 
                                                  output_size=np.shape(self.test_output)[1],
                                                  hidden_layers=hyperparameters['hidden_layers'],
                                                  dropout=hyperparameters['dropout'],
                                                  output_activation=hyperparameters['output_activation'],
                                                  batch_norm=hyperparameters['batch_norm'],
                                                  activation=hyperparameters['activation'])
        
        forward_dnn.load_state_dict(torch.load(f'forward_DNN/forwardDNN.pth'))
        forward_dnn.eval()
        
        return forward_dnn
    
    
    def load_scalers(self):

        input_scaler = joblib.load(f'{self.scalers_path}/input_scaler_inverse.pkl')
        output_scaler = joblib.load(f'{self.scalers_path}/output_scaler_inverse.pkl')

        return input_scaler, output_scaler
    
    def get_predictions(self):
        
        inverseDNN = self.load_inverse_dnn()
        input_scaler, output_scaler = self.load_scalers()
        
        test_output_scaled = input_scaler.transform(self.test_output)
        test_output_torch = torch.tensor(test_output_scaled, dtype=torch.float32).to(device)
        
        inverse_predictions = inverseDNN(test_output_torch)
        inverse_predictions_scaled = output_scaler.inverse_transform(inverse_predictions.detach().cpu().numpy())
        
        prediction_output = self.evaluate(inverse_predictions_scaled)
    
        return prediction_output
    
    def get_predictions_forward(self):
        
        forwardDNN = self.load_forward_dnn()
        output_scaler, input_scaler = self.load_scalers()
        
        test_input_scaled = input_scaler.transform(self.test_input)
        test_input_torch = torch.tensor(test_input_scaled, dtype=torch.float32).to(device)
        
        forward_predictions = forwardDNN(test_input_torch)
        prediction_output = output_scaler.inverse_transform(forward_predictions.detach().cpu().numpy())
           
        return prediction_output
        
    def error_metrics(self):
        
        prediction_output = self.get_predictions()
        
        r2 = r2_score(self.test_output, prediction_output)
        rmse = np.sqrt(mean_squared_error(self.test_output, prediction_output))
        mape = mean_absolute_percentage_error(self.test_output, prediction_output)
        nmax_ae = np.mean(np.max(np.abs(self.test_output - prediction_output), axis=0)/np.max(np.abs(self.test_output - np.mean(self.test_output, axis=0))))
        
        return r2, rmse, mape, nmax_ae
    
    def error_metrics_forward(self):
        
        prediction_output = self.get_predictions_forward()
        
        r2 = r2_score(self.test_output, prediction_output)
        rmse = np.sqrt(mean_squared_error(self.test_output, prediction_output))
        mape = mean_absolute_percentage_error(self.test_output, prediction_output)
        nmax_ae = np.mean(np.max(np.abs(self.test_output - prediction_output), axis=0)/np.max(np.abs(self.test_output - np.mean(self.test_output, axis=0))))

        
        return r2, rmse, mape, nmax_ae


        