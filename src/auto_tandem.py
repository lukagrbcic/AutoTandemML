import numpy as np
import sys
import joblib

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
                 verbose=1, forward_model=None):
        
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
        self.forward_model = forward_model
    
    
    def get_foward_model(self):
        
        run = al.activeLearner(self.f, self.lb, self.ub,
                                self.init_size, self.batch_size,
                                self.max_samples, self.sampler,
                                self.algorithm, self.test_data,
                                verbose=self.verbose, return_model=True, return_hf_samples=True)
        
        _, model, X_hf, y_hf = run.run()
        
        joblib.dump(model, 'forward_model.pkl')
        np.save('x_hf.npy', X_hf)
        np.save('y_hf.npy', y_hf)
        
        return model, X_hf, y_hf
    
    def get_lf_samples(self, model):

        X = samplers('lhs', self.lf_samples, self.lb, self.ub, self.algorithm).generate_samples()
        y = model.predict(X)
        
        return X, y
    
    def get_inverse_model(self):
        
        self.forward_model, X_hf, y_hf = self.get_foward_model()

        X, y = self.get_lf_samples(self.forward_model)
        
        
        
        
        
        
        