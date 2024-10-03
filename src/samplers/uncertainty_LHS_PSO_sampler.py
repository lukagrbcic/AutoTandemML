import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from indago import PSO, DE
import random
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
# np.random.seed(random.randint(1,1209423))




class uncertaintyLHSPSOSampler:
    
    def __init__(self, model, sample_size, lb, ub, algorithm, penalty_factor=1, c=1):
        
        self.model = model
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
        self.algorithm = algorithm
        self.c = c
        self.penalty_factor = penalty_factor
        
    def get_unc_samples(self):

        X = []
        f = []
        samples = qmc.scale(qmc.LatinHypercube(d=len(self.lb)).random(n=len(self.lb)*self.sample_size), self.lb, self.ub)

        for s in samples:
            def uncertainty(x):
                
                p = []
                if len(X) > 0:
                    
                    for convbest in X:
                        val = np.linalg.norm(convbest - x)
                        p.append(1/val)    
       
         
                if self.algorithm == 'rf':
                    tree_preds = np.array([tree.predict(x.reshape(1,-1)) for tree in self.model.estimators_])
                    std = np.sum(np.std(tree_preds, axis=0))
        
                return -std + self.penalty_factor*np.sum(p)

            
            x0 = np.random.normal(loc=s, scale=0.2*s, size=(3, np.shape(samples)[1]))
            x0 = np.clip(x0, self.lb, self.ub)
                           
            optim = PSO()
            optim.params['swarm_size'] = 3
            optim.params['cognitive_rate'] = 2
            optim.params['social_rate'] = 1
            optim.X0 = x0
            optim.max_evaluations = 100 #6*len(s)
            optim.lb = self.lb
            optim.ub = self.ub
            optim.evaluation_function = uncertainty 
            run = optim.optimize()
            min_x = run.X
            min_f = run.f    
                
            X.append(min_x)
            f.append(min_f)
            
        # f = np.concatenate(f)
        # X = np.array(X)
        
        
        
                    
        f = np.array(f)
        X = np.array(X)
        
        X_f = np.hstack((X, f.reshape(-1,1)))
        cluster = KMeans(n_clusters=self.sample_size, n_init='auto').fit(X_f)

        X = cluster.cluster_centers_[:, :-1]
        
                       
        return X
            
            
        
        