import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import cdist

class bcSampler:
    
    def __init__(self, sample_size, lb, ub):
        
        self.sample_size = sample_size
        self.lb = lb
        self.ub = ub
        
    def gen_BC_samples(self, scale=2, maxCand=100):

        
        first_sample = np.random.uniform(self.lb, self.ub, size=(1, len(self.lb)))
        selected_samples = [first_sample]  
    
        for i in range(2, self.sample_size + 1):

            nCand = min(scale * i, maxCand)
    
            candidates = np.random.uniform(self.lb, self.ub, size=(nCand, len(self.lb)))

            selected_array = np.vstack(selected_samples)
    
            distances = cdist(candidates, selected_array, metric='euclidean')  # Shape: (nCand, i-1)
    
            min_distances = np.min(distances, axis=1)
    
            idx_best_candidate = np.argmax(min_distances)
            best_candidate_sample = candidates[idx_best_candidate]
            
            selected_samples.append(best_candidate_sample.reshape(1, len(self.lb)))
    
        selected_samples_array = np.vstack(selected_samples)
        
        return selected_samples_array
    
    
    
