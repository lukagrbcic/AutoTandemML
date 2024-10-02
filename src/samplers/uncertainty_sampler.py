import numpy as np
from scipy.optimize import minimize
from indago import PSO



class uncertaintySampler:
    
    def __init__(self, model, sample_size, lb, ub, algorithm):
        
        self.model = model
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
        self.algorithm = algorithm
        
    def get_unc_samples(self):

        X = []
        f = []
        for i in range(self.sample_size):
            
            
            
            
            def uncertainty(x):
    
                if self.algorithm == 'rf':
                    tree_preds = np.array([tree.predict(x.reshape(1,-1)) for tree in self.model.estimators_])
                    std = np.std(tree_preds, axis=0)
                return -np.sum(std)
                       
            # dim = len(self.lb)
            # min_f = 1
            # min_x = None
            
            # x0 = np.random.uniform(self.lb, self.ub, (1, dim))[0]
            # bounds = np.array([[self.lb[i], self.ub[i]] for i in range(len(self.lb))])
            
            # res = minimize(uncertainty, x0=x0, bounds=bounds, method='L-BFGS-B')
            # if res.fun < min_f:
            #     min_f = res.fun
            #     min_x = res.x
            
            optimizer = PSO()
            optimizer.evaluation_function = uncertainty 
            optimizer.lb = self.lb
            optimizer.ub = self.ub
            optimizer.max_evaluations = 80
            result = optimizer.optimize()
            min_x = result.X 
            min_f = result.f
                
            X.append(min_x)
            f.append(min_f)
                                
        X = np.array(X)
        f = np.array(f)
    
        return X
            
            
        
        