import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from indago import PSO
import random
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

from goal_functions import goal_function


class modelLHSSampler:
    def __init__(self, model, sample_size, lb, ub, algorithm, 
                 function='uncertainty', x_sampled=[], clustering=False):

        
        self.model = model
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
        self.algorithm = algorithm
        self.function = function
        self.x_sampled = x_sampled
        self.clustering = False
        
    def get_samples(self):

        X = []
        f = []
        samples = qmc.scale(qmc.LatinHypercube(d=len(self.lb)).random(n=20*self.sample_size), self.lb, self.ub)
        for s in samples:
            def get_values(x):
                
                # p = []
                # if len(self.x_sampled) > 0:
                #     for convbest in self.x_sampled:
                #         val = np.linalg.norm(convbest - x)
                #         p.append((1/val)**2)  
                        
                preds = np.concatenate(np.array([model.predict([x]) for model in self.model.estimators_]))
                                    
                value = goal_function(method=self.function).calculate(preds)
                
                return value + value*np.sum(p)

            
            # x0 = np.random.normal(loc=s, scale=0.2*s, size=(3, np.shape(samples)[1]))
            # x0 = np.clip(x0, self.lb, self.ub)
                           
            optim = PSO()
            # optim.params['swarm_size'] = 3
            # optim.params['cognitive_rate'] = 2
            # optim.params['social_rate'] = 1
            optim.X0 = s
            optim.max_evaluations = 30 #6*len(s)
            optim.lb = self.lb
            optim.ub = self.ub
            optim.evaluation_function = get_values 
            run = optim.optimize()
            min_x = run.X
            min_f = run.f    
                
            X.append(min_x)
            f.append(min_f)
            
      
        f = np.array(f)
        if self.clustering == False:
            X = np.array(X)       
            X = X[np.argsort(f)[:self.sample_size]]
        
        else:
            x_f = np.hstack((X, f.reshape(-1,1)))
            cluster = KMedoids(n_clusters=self.sample_size).fit(x_f)
            X = cluster.cluster_centers_[:, :-1]
       
        return X
            
            
        
        