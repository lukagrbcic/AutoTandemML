import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from .lhs_sampler import lhsSampler
from .poisson_sampler import poissonSampler
from .random_sampler import randomSampler
from .greedyfp_sampler import greedyFPSampler
from .bc_sampler import bcSampler


from .model_sampler import modelSampler
from .model_greedy_sampler import modelGFPSampler
from .modelHC_sampler import modelHCSampler
from .modelLHS_sampler import modelLHSSampler


class samplers:
    
    def __init__(self, sampler, batch_size, lb, ub, algorithm, sampled_points=[], model=None):
        
        self.sampler = sampler
        self.model = model
        self.batch_size = batch_size
        self.lb = lb
        self.ub = ub
        self.algorithm = algorithm
        self.sampled_points = sampled_points
    
    def generate_samples(self, seed=None):
        
        if self.sampler == 'lhs':
            
            X = lhsSampler(self.batch_size, self.lb, self.ub).gen_LHS_samples()
            
        elif self.sampler == 'random':
            
            X = randomSampler(self.batch_size, self.lb, self.ub).gen_random_samples()

        elif self.sampler == 'greedyfp':
            
            X = greedyFPSampler(self.batch_size, self.lb, self.ub).gen_GFP_samples()

        elif self.sampler == 'bc':
            
            X = bcSampler(self.batch_size, self.lb, self.ub).gen_BC_samples()
        
        elif self.sampler == 'poisson':
            
            X = poissonSampler(self.batch_size, self.lb, self.ub).gen_poisson_samples()
        
        
        elif self.sampler.split('_')[0] == 'model':
            
            X = modelSampler(self.model, self.batch_size, 
                                self.lb, self.ub, self.algorithm[0],
                                self.sampler.split('_')[-1], self.sampled_points).get_samples()
        
        elif self.sampler.split('_')[0] == 'modelgreedy':
            
            X = modelGFPSampler(self.model, self.batch_size, 
                                self.lb, self.ub, self.algorithm[0],
                                self.sampler.split('_')[-1], self.sampled_points).get_samples()
        
        
        elif self.sampler.split('_')[0] == 'modelHC':
            
            X = modelHCSampler(self.model, self.batch_size, 
                                    self.lb, self.ub, self.algorithm[0], 
                                    function=self.sampler.split('_')[-1], 
                                    x_sampled=self.sampled_points).get_samples()
            
        elif self.sampler.split('_')[0] == 'modelLHS':
            
            X = modelLHSSampler(self.model, self.batch_size, self.lb, self.ub, self.algorithm[0], 
                                     self.sampler.split('_')[-1],
                                     self.sampled_points).get_samples()
            
        elif self.sampler.split('_')[0] == 'ensemble':
            
            X_1 = modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'uncertainty', self.sampled_points).get_samples()
            
            X_2 = modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'entropy', self.sampled_points).get_samples()
            
            X_3 = modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'quantile', self.sampled_points).get_samples()
            
            size = int(self.batch_size/3)
            
            X = np.vstack((X_1[:size, :], X_2[:size, :], X_3[:size, :]))
        

        elif self.sampler.split('_')[0] == 'ensemble_cluster':
            
            X_1 = modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'uncertainty', self.sampled_points).get_samples()
            
            X_2 = modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'entropy', self.sampled_points).get_samples()
            
            X_3 = modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'quantile', self.sampled_points).get_samples()
            
            scaler = StandardScaler()
            
            X_stacked = np.vstack((X_1, X_2, X_3))
            
            X_stacked = scaler.fit_transform(X_stacked)
            
            cluster = KMeans(n_clusters=self.batch_size, n_init='auto').fit(X_stacked)
            X = cluster.cluster_centers_
            
            X = scaler.inverse_transform(X)
                
        return X
        
    
    
    
        
