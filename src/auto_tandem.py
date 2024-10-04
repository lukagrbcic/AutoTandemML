import numpy as np
import sys

from ensemble_regressor import EnsembleRegressor
import active_learning as al

sys.path.insert(0, 'samplers')
sys.path.insert(1, 'models')

from generate_samples import samplers



class AutoTNN:
    
    def __init__(self, f, lb, ub,
                 init_size, 
                 batch_size,
                 max_samples, 
                 algorithm,
                 test_data,
                 lf_samples=0,
                 sampler='model_uncertainty',
                 verbose=0):
        
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
    
    
    def get_foward_model(self):
        
        run = al.activeLearner(self.f, self.lb, self.ub,
                                self.init_size, self.batch_size,
                                self.max_samples, self.sampler,
                                self.algorithm, self.test_data,
                                verbose=self.verbose, return_model=True)
        _, model = run.run()
        
        return model
    
    def get_lf_samples(self, model):

        X = samplers('lhs', self.lf_samples, self.lb, self.ub, self.algorithm).generate_samples()
        y = model.predict(X)
        
        return X, y
    
    def get_inverse_model(self):
        
        forward_model = self.get_foward_model()
        X, y = self.get_lf_samples(forward_model)
        
        
        
        
        
        