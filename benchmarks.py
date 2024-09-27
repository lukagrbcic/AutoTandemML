import numpy as np
import joblib
import sys

sys.path.insert(0,'models')




class load_model:
    
    def __init__(self, f_name):
        
        self.f_name = f_name
        
    def load_model(self):
        
        if self.f_name == 'inconel_benchmark':
            
            ml_model = joblib.load('models/inconel_model.pkl')
            pca = joblib.load('models/inconel_pca.pkl')
            model = (pca, ml_model)    
            
        return model

class benchmark_functions:
    
    def __init__(self, f_name, model):
        
        self.f_name = f_name
        self.model = model

    def get_bounds(self):
        
        if self.f_name == 'inconel_benchmark':
            
            lb = np.array([0.3, 10, 15])
            ub = np.array([1.2, 700, 28])
        
        return lb, ub
    
            
    def inconel_benchmark(self, x):
        
        pca, ml_model = self.model
        
        f = pca.inverse_transform(ml_model.predict(x))
        
        return  f
    
    
    def evaluate(self, x):

        if self.f_name == 'inconel_benchmark':
            responses = self.inconel_benchmark(x)
        
        return responses




# name = 'inconel_benchmark'
# model = load_model(name).load_model()

# f = benchmark_functions(name, model)
# lb, ub = f.get_bounds()

# def evaluation_function(x):
    
#     value = f.evaluate(x)
    
#     return value


# emissivity = evaluation_function([[1, 100, 16], [0.5, 200, 27]])





