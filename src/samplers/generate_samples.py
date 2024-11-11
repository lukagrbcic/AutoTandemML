import lhs_sampler as lhs
import poisson_sampler as poisson
import random_sampler as rnd
import greedyfp_sampler as gfp
import bc_sampler as bc

import model_sampler as ms
import modelHC_sampler as mhcs
import modelLHS_sampler as mlhs

import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler, MinMaxScaler



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
            
            X = lhs.lhsSampler(self.batch_size, self.lb, self.ub).gen_LHS_samples()
            
        elif self.sampler == 'random':
            
            X = rnd.randomSampler(self.batch_size, self.lb, self.ub).gen_random_samples()

        elif self.sampler == 'greedyfp':
            
            X = gfp.greedyFPSampler(self.batch_size, self.lb, self.ub).gen_GFP_samples()

        elif self.sampler == 'bc':
            
            X = bc.bcSampler(self.batch_size, self.lb, self.ub).gen_BC_samples()
        
        elif self.sampler == 'poisson':
            
            X = poisson.poissonSampler(self.batch_size, self.lb, self.ub).gen_poisson_samples()
        
        
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
            
            X_1 = ms.modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'uncertainty', self.sampled_points).get_samples()
            
            X_2 = ms.modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'entropy', self.sampled_points).get_samples()
            
            X_3 = ms.modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'quantile', self.sampled_points).get_samples()
            
            size = int(self.batch_size/3)
            
            X = np.vstack((X_1[:size, :], X_2[:size, :], X_3[:size, :]))
        

        elif self.sampler.split('_')[0] == 'ensemble_cluster':
            
            X_1 = ms.modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'uncertainty', self.sampled_points).get_samples()
            
            X_2 = ms.modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'entropy', self.sampled_points).get_samples()
            
            X_3 = ms.modelSampler(self.model, self.batch_size, 
                                  self.lb, self.ub, self.algorithm[0],
                                  'quantile', self.sampled_points).get_samples()
            
            scaler = StandardScaler()
            
            X_stacked = np.vstack((X_1, X_2, X_3))
            
            X_stacked = scaler.fit_transform(X_stacked)
            
            cluster = KMeans(n_clusters=self.batch_size, n_init='auto').fit(X_stacked)
            X = cluster.cluster_centers_
            
            X = scaler.inverse_transform(X)
                
        return X
        
    
    
    
        