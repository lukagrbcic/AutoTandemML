import numpy as np
import sys
import joblib

from ensemble_regressor import EnsembleRegressor
import active_learning as al


sys.path.insert(0, 'samplers')
sys.path.insert(1, 'models')

from generate_samples import samplers
from sklearn.metrics import *
from optimize_inverse import get_hyperparameters
from DNNRegressor import TorchDNNRegressor
from model_factory import ModelFactory
from get_forward import forwardDNN
from get_inverse import inverseDNN
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(random.randint(1,100000))

class AutoTNN:
    
    def __init__(self, f, lb, ub,
                 init_size, 
                 batch_size,
                 max_samples, 
                 algorithm,
                 test_data,
                 lf_samples=0,
                 sampler='model_uncertainty',
                 verbose=0, combinations=100, 
                 forward_param_dist=None, 
                 inverse_param_dist=None,
                 forward_model=None,
                 x_init=[],
                 y_init=[]):
        
        self.f = f
        self.lb = lb
        self.ub = ub
        self.init_size = init_size
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.algorithm = algorithm
        self.test_data = test_data
        self.lf_samples = lf_samples
        self.sampler = sampler
        self.verbose = verbose
        self.combinations = combinations
        self.forward_param_dist = forward_param_dist
        self.inverse_param_dist = inverse_param_dist
        self.forward_model = forward_model
        self.x_init = x_init
        self.y_init = y_init
    
    
    def get_foward_model(self):
        
        run = al.activeLearner(self.f, self.lb, self.ub,
                                self.init_size, self.batch_size,
                                self.max_samples, self.sampler,
                                self.algorithm, self.test_data,
                                verbose=0, return_model=True, return_hf_samples=True)
        
        _, model, X_hf, y_hf = run.run()

        return model, X_hf, y_hf
    
    def get_lf_samples(self, model):

        X = samplers('lhs', self.lf_samples, self.lb, self.ub, self.algorithm).generate_samples()
        y = model.predict(X)
        
        return X, y
    
    def get_forward_DNN(self, X, y):
        
        
        if self.forward_param_dist is None:
        
            forward_param_dist = {
                'model_type': ['mlp'],
                'hidden_layers': [[64], [128], [256], [128, 128],
                                  [256, 256], [512, 512], [64, 128, 64],
                                  [128, 256, 128], [256, 512, 256],
                                  [64, 128, 256, 128, 64]],
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
        
        else: forward_param_dist = self.forward_param_dist
    
        fwd_hyperparameters = get_hyperparameters(X, y, forward_param_dist, n_iter=self.combinations).run()
        forwardDNN(X, y, fwd_hyperparameters).train_save()
        
        return fwd_hyperparameters
        
    
    def get_inverse_DNN(self):
        
        if self.verbose == True:
            
            print ('Generating dataset')
        
        if len(self.x_init) == 0:    
            self.forward_model, X_hf, y_hf = self.get_foward_model()
        else: 
            X_hf = self.x_init
            y_hf = self.y_init
        
        if self.lf_samples > 0:
            
            X_lf, y_lf = self.get_lf_samples(self.forward_model)
            X_hf_init = np.vstack((X_hf, X_lf))
            y_hf_init = np.vstack((y_hf, y_lf))
            
            indices = np.random.permutation(X_hf_init.shape[0])
            X_hf = X_hf_init[indices]
            y_hf = y_hf_init[indices]
        
        if self.verbose == True:
            
            print ('Optimizing and training forward DNN')
            
        fwd_hyperparameters = self.get_forward_DNN(X_hf, y_hf)
        
        if self.verbose == True:
            
            print ('Optimizing inverse DNN')
            
        if self.inverse_param_dist is None:
            
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
            
        else:
            
            param_dist = self.inverse_param_dist
        
        
        self.inverse_hyperparameters = get_hyperparameters(y_hf, X_hf, 
                                        param_dist, seed=np.random.randint(1,10000), n_iter=self.combinations, 
                                        forward_model_hyperparameters=fwd_hyperparameters).run()
        
        np.save('inverseDNN/model_config.npy', self.inverse_hyperparameters)
        
        if self.verbose == True:
            
            print ('Training inverse DNN')
            
        inverseDNN(y_hf, X_hf, self.inverse_hyperparameters, 
                   forward_model_hyperparameters=fwd_hyperparameters, verbose=False).train_save()
            
        
        


        
        
        
        