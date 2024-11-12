import numpy as np
from scipy.optimize import minimize
from indago import PSO
from goal_functions import goal_function
import random
from scipy.spatial.distance import cdist

np.random.seed(random.randint(0, 10223))



class modelGFPSampler:
    
    def __init__(self, model, sample_size, lb, ub, algorithm, function='uncertainty', x_sampled=[]):
        
        self.model = model
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
        self.algorithm = algorithm
        self.function = function     
        self.x_sampled = x_sampled
        # self.lambda_ = lambda_

    def get_samples(self):

        X = []
        f = []

        def get_values(x):               
  
            preds = np.concatenate(np.array([model.predict([x]) for model in self.model.estimators_]))
            
            value = goal_function(method=self.function).calculate(preds)
            
            X.append(x)
            f.append(value)

            return value
        
                      
        optimizer = PSO()
        optimizer.evaluation_function = get_values 
        optimizer.lb = self.lb
        optimizer.ub = self.ub
        optimizer.max_evaluations = 3*self.sample_size
        # optimizer.monitoring = 'basic'
        result = optimizer.optimize()
        min_x = result.X 
        min_f = result.f
                        
      
        X = np.array(X)
        f = np.array(f)
        
        f_indx = np.argsort(f)[:2*self.sample_size]
        f = f[f_indx]
        X = X[f_indx]
        
        
        selected_indices = []
        available_indices = list(range(len(X)))
    
        first_idx = np.argmin(f)
        selected_indices.append(first_idx)
        available_indices.remove(first_idx)
    
        for _ in range(self.sample_size - 1):
            selected_samples = X[selected_indices]  
    
            remaining_candidates = X[available_indices]  
    
            distances = cdist(remaining_candidates, selected_samples, metric='euclidean')
    
            min_distances = np.min(distances, axis=1)
    
            idx_in_remaining = np.argmax(min_distances)
            idx_max_min_dist = available_indices[idx_in_remaining]
    
            selected_indices.append(idx_max_min_dist)
            available_indices.remove(idx_max_min_dist)
    
        selected_samples = X[selected_indices]
        
    
        return X
            
            
        
        
