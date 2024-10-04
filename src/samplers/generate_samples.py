import lhs_sampler as lhs
import random_sampler as rnd
import model_sampler as ms
import modelHC_sampler as mhcs
import modelLHS_sampler as mlhs



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

        return X
        
    
    
    
        