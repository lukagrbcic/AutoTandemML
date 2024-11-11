import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import cdist

class greedyFPSampler:
    
    def __init__(self, sample_size, lb, ub):
        
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
            
    def gen_GFP_samples(self, scale=2):

        M = self.sample_size * scale  
        
        candidates = np.random.uniform(self.lb, self.ub, size=(M, len(self.lb)))    
        
        
        selected_indices = []
        available_indices = list(range(M))
    
        first_idx = np.random.choice(available_indices)
        selected_indices.append(first_idx)
        available_indices.remove(first_idx)
    
        for _ in range(self.sample_size - 1):
            selected_samples = candidates[selected_indices]  
    
            remaining_candidates = candidates[available_indices]  
    
            distances = cdist(remaining_candidates, selected_samples, metric='euclidean')
    
            min_distances = np.min(distances, axis=1)
    
            idx_in_remaining = np.argmax(min_distances)
            idx_max_min_dist = available_indices[idx_in_remaining]
    
            selected_indices.append(idx_max_min_dist)
            available_indices.remove(idx_max_min_dist)
    
        selected_samples = candidates[selected_indices]
        
        return selected_samples
    
    

