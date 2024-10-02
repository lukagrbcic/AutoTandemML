import lhs_sampler as lhs
import random_sampler as rnd
import uncertainty_sampler as unc 
import uncertainty_LHS_PSO_sampler as unc_lhs_pso
import uncertainty_LHS_PSO_HC_sampler as unc_hc



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
        
        elif self.sampler == 'unc':
            X = unc.uncertaintySampler(self.model, self.batch_size, self.lb, self.ub, self.algorithm[0]).get_unc_samples()
        
        elif self.sampler == 'unc_lhs_pso':
            X = unc_lhs_pso.uncertaintyLHSPSOSampler(self.model, self.batch_size, self.lb, self.ub, self.algorithm[0]).get_unc_samples()
                    
        elif self.sampler == 'unc_hc':
            X = unc_hc.uncertaintyHCSampler(self.model, self.batch_size, self.lb, self.ub, self.algorithm[0], self.sampled_points).get_unc_samples()

        return X
        
    
    
    
        