import numpy as np
import random

np.random.seed(random.randint(0, 10223))


class randomSampler:
    
    def __init__(self, sample_size, lb, ub):
        
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
        
    def gen_random_samples(self):
        
        n = self.sample_size
        d = len(self.lb)
        sample_set = np.array([np.random.uniform(self.lb, self.ub, d) for i in range(n)])
        
        return sample_set
    
        