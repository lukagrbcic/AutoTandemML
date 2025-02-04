import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import RandomizedSearchCV
import sklearn.metrics
import numpy as np


from .DNNRegressor import TorchDNNRegressor


class get_hyperparameters:
    def __init__(self, X, y, param_dist, n_iter=10, cv=3, 
                 scoring='neg_root_mean_squared_error', n_jobs=4,
                 verbose=0, criterion='rmse', seed=11, forward_model_hyperparameters=None):
        
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
        self.forward_model_hyperparameters = forward_model_hyperparameters
        
        
    def run(self):
        
        model = TorchDNNRegressor(input_size=np.shape(self.X)[1],
                                  output_size=np.shape(self.y)[1], 
                                  verbose=False, criterion=self.criterion, 
                                  forward_model_hyperparameters=self.forward_model_hyperparameters)
        
        random_search = RandomizedSearchCV(model, param_distributions=self.param_dist, 
                                            n_iter=self.n_iter, cv=self.cv, verbose=self.verbose, random_state=self.seed, 
                                            scoring=self.scoring, n_jobs=self.n_jobs)
        
        random_search.fit(self.X, self.y)
        best_params = random_search.best_params_
        
        return best_params

