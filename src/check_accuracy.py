import numpy as np
from sklearn.metrics import mean_absolute_percentage_error


"""
https://www.cirm-math.fr/ProgWeebly/Renc1762/Welch.pdf

source for worst error 
"""

class error:
    
    def __init__(self, model, test_data):
        
        self.model = model
        self.test_data = test_data #(inputs, outputs)
    
    @staticmethod
    def normalized_rmse(y_true, y_pred):
        mse = np.mean((y_true - y_pred)**2, axis=0)
        rmse = np.sqrt(mse)
        range_y = np.max(y_true, axis=0) - np.min(y_true, axis=0)
        nrmse = rmse / range_y
        return np.mean(nrmse)

    @staticmethod
    def max_normalized_rmse(y_true, y_pred):
        mse = np.mean((y_true - y_pred)**2, axis=0)
        rmse = np.sqrt(mse)
        range_y = np.max(y_true, axis=0) - np.min(y_true, axis=0)
        nrmse = rmse / range_y
        return np.max(nrmse)

    @staticmethod
    def normalized_max_ae(y_true, y_pred):
        abs_diff = np.abs(y_true - y_pred)
        abs_diff_mean = np.abs(y_true - np.mean(y_true))
        nae = np.max(abs_diff)/np.max(abs_diff_mean)
        return nae
    
    @staticmethod  
    def mape(y_true, y_pred):
        val = np.mean(np.array([mean_absolute_percentage_error(y_true[i], y_pred[i]) for i in range(len(y_pred))]))
        return val
           
    def test_set(self):
        
        test_inputs, test_outputs = self.test_data        
        test_predictions = self.model.predict(test_inputs)
        
        nrmse = self.normalized_rmse(test_outputs, test_predictions)
        max_nrmse = self.max_normalized_rmse(test_outputs, test_predictions)
        max_nae = self.normalized_max_ae(test_outputs, test_predictions)
        mape_ = self.mape(test_outputs, test_predictions)

        return nrmse, max_nrmse, max_nae, mape_
        
