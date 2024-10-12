import numpy as np

class goal_function:
    
    def __init__(self, method='uncertainty'):
        
        self.method = method
        self.alfa = 0.5
   
    def uncertainty(self, preds):
                
        total_uncertainty = -np.sum(np.std(preds, axis=0))
        
        return total_uncertainty
    
    def uncertainty_quantile(self, preds):
        
        q02 = np.percentile(preds, 0.02, axis=0)
        q98 = np.percentile(preds, 0.98, axis=0)
        
        diff = np.abs(q98 - q02)
        total_uncertainty = -np.sum(diff)
        
        return total_uncertainty 

    def entropy(self, preds):
        
        mu = np.mean(preds, axis=0)  
        sigma_squared = np.var(preds, axis=0)
        entropy = 0.5 * np.log(2 * np.pi * np.e * sigma_squared)
        total_entropy = -np.sum(entropy)
        
        return total_entropy
    
    def mixed(self, preds):
        
        total_uncertainty = self.uncertainty(preds) 
        total_entropy = self.entropy(preds)
        total_quantile = self.uncertainty_quantile(preds)
        
        return 0.33*total_entropy + 0.33*total_uncertainty + 0.33*total_quantile
        
    def calculate(self, preds):
        
        if self.method == 'uncertainty':
            
            value = self.uncertainty(preds)

        if self.method == 'quantile':
            
            value = self.uncertainty_quantile(preds)
        
        if self.method == 'entropy':
            
            value = self.entropy(preds)
        
        if self.method == 'mixed':
            
            value = self.mixed(preds)
            
        return value
        
        