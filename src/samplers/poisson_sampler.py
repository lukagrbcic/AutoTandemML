import numpy as np
from scipy.stats import qmc

class poissonSampler:
    
    def __init__(self, sample_size, lb, ub):
        
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
        
    def gen_poisson_samples(self, seed=None):
        
        n = self.sample_size
        d = len(self.ub)
        
        if seed == None:
            sampler = qmc.PoissonDisk(d=d, ncandidates=n)
        else: 
            sampler = qmc.PoissonDisk(d=d, ncandidates=n, seed=seed)
            
        samples = sampler.random(n=n)
        sample_set = qmc.scale(samples, self.lb, self.ub)
        
        return sample_set
    
    
    
    
        
