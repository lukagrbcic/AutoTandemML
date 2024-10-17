import numpy as np
from scipy.optimize import minimize
from indago import PSO
from goal_functions import goal_function
import random
np.random.seed(random.randint(1,10000))



class modelSampler:
    
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
        for i in range(self.sample_size):
            
            def get_values(x):               
                p = []
                if len(self.x_sampled) > 0:
                    for convbest in self.x_sampled:
                        val = np.linalg.norm(convbest - x)
                        p.append(1/val)  
      
                preds = np.concatenate(np.array([model.predict([x]) for model in self.model.estimators_]))
                
                value = goal_function(method=self.function).calculate(preds)
                
                return value + value*np.sum(p)
            
 
            # xs = []
            # fs = []
            
            # for i in range(80):
            #     dim = len(self.lb)
            #     min_f = np.inf
            #     min_x = None
                
                
            #     x0 = np.random.uniform(self.lb, self.ub, (1, dim))[0]
            #     bounds = np.array([[self.lb[i], self.ub[i]] for i in range(len(self.lb))])
            #     res = minimize(get_values, x0=x0, bounds=bounds, method='L-BFGS-B')
            #     if res.fun < min_f:
            #         min_f = res.fun
            #         min_x = res.x
                
            #     xs.append(min_x)
            #     fs.append(min_f)
                
            # min_f_index = np.argmin(fs)
            # min_f = fs[min_f_index]
            # min_x = xs[min_f_index]
                           
            optimizer = PSO()
            optimizer.evaluation_function = get_values 
            optimizer.lb = self.lb
            optimizer.ub = self.ub
            optimizer.max_evaluations = 30
            result = optimizer.optimize()
            min_x = result.X 
            min_f = result.f
                            
            X.append(min_x)
            f.append(min_f)
                                
        X = np.array(X)
        f = np.array(f)
        
        X = X[np.argsort(f)]
    
        return X
            
            
        
        