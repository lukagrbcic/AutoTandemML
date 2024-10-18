import lhs_sampler as lhs
import random_sampler as rnd
import model_sampler as ms
import modelHC_sampler as mhcs
import modelLHS_sampler as mlhs

import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids



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
            
            X = lhs.lhsSampler(self.batch_size, self.lb, self.ub).gen_LHS_samples(seed)
            
        elif self.sampler == 'random':
            
            X = rnd.randomSampler(self.batch_size, self.lb, self.ub).gen_random_samples()
        
        elif self.sampler.split('_')[0] == 'model':
            
            X = ms.modelSampler(self.model, self.batch_size, 
                                self.lb, self.ub, self.algorithm[0],
                                self.sampler.split('_')[-1], self.sampled_points).get_samples()
        
        elif self.sampler.split('_')[0] == 'modelHC':
            
            X = mhcs.modelHCSampler(self.model, self.batch_size, 
                                    self.lb, self.ub, self.algorithm[0], 
                                    function=self.sampler.split('_')[-1], 
                                    x_sampled=self.sampled_points).get_samples()
            
        elif self.sampler.split('_')[0] == 'modelLHS':
            
            X = mlhs.modelLHSSampler(self.model, self.batch_size, self.lb, self.ub, self.algorithm[0], 
                                     self.sampler.split('_')[-1],
                                     self.sampled_points).get_samples()
            
        elif self.sampler.split('_')[0] == 'ensemble':
            
            X_1 = mhcs.modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'uncertainty', self.sampled_points).get_samples()
            
            # X_2 = mhcs.modelHCSampler(self.model, self.batch_size, 
            #                         self.lb, self.ub, self.algorithm[0], 
            #                         function='quantile', 
            #                         x_sampled=self.sampled_points).get_samples()
            
            # X_3 = mlhs.modelLHSSampler(self.model, self.batch_size, 
            #                            self.lb, self.ub, self.algorithm[0], 
            #                          'quantile',
            #                          self.sampled_points).get_samples()
            
            X_2 = mhcs.modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'entropy', self.sampled_points).get_samples()
            
            X_3 = mhcs.modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'quantile', self.sampled_points).get_samples()
            
            size = int(self.batch_size/3)
            
            X = np.vstack((X_1[:size, :], X_2[:size, :], X_3[:size, :]))

        elif self.sampler.split('_')[0] == 'ensemble_cluster':
            
            X_1 = mhcs.modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'uncertainty', self.sampled_points).get_samples()
            
            X_2 = mhcs.modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'entropy', self.sampled_points).get_samples()
            
            X_3 = mhcs.modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'quantile', self.sampled_points).get_samples()
            
            
            X_stacked = np.vstack((X_1, X_2, X_3))
            
            cluster = KMedoids(n_clusters=self.batch_size).fit(X_stacked)
            X = cluster.cluster_centers_
                
        return X
        
    
    
    
        