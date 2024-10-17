import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import RandomizedSearchCV
import sklearn.metrics
import numpy as np
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

material = 'airfoil_Re_3_6'
X_train = np.load(f'../data/{material}_data/input_train_data.npy')#[:3000]
y_train = np.load(f'../data/{material}_data/output_train_data.npy')#[:3000]

class get_hyperparameters:
    def __init__(self, X, y, param_dist, n_iter=100, cv=3, 
                 scoring='neg_root_mean_squared_error', n_jobs=1, verbose=2, criterion='mse', seed=11):
        
        self.X = X
        self.y = y
        self.param_dist = param_dist
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.criterion = criterion
        self.seed = seed
        
        
    def run(self):
        
        model = TorchDNNRegressor(input_size=np.shape(X_train)[1],
                                  output_size=np.shape(y_train)[1], verbose=False, criterion=self.criterion)
        
        random_search = RandomizedSearchCV(model, param_distributions=self.param_dist, 
                                            n_iter=self.n_iter, cv=self.cv, verbose=self.verbose, random_state=self.seed, 
                                            scoring=self.scoring, n_jobs=self.n_jobs)
        
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        
        print("Best parameters found: ", best_params)

        return best_params

hyperparams = get_hyperparameters(X_train, y_train, param_dist).run()