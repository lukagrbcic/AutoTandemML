import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

"""
https://www.cirm-math.fr/ProgWeebly/Renc1762/Welch.pdf

source for nmax_ae

https://wires.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/widm.1157
https://arxiv.org/pdf/1907.01181

"""

class error:
    
    def __init__(self, model, test_data):
        
        self.model = model
        self.test_data = test_data #(inputs, outputs)
        
        test_inputs, test_outputs = self.test_data        

        
    @staticmethod
    def rmse(y_true, y_pred):
        
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mape(y_true, y_pred):
        
        return mean_absolute_percentage_error(y_true, y_pred)
    
    @staticmethod
    def range_nrmse(y_true, y_pred):
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
        range_ = np.ptp(y_true, axis=0)
        normalized_rmse = np.mean(rmse/range_)
        
        return normalized_rmse
    
    @staticmethod
    def max_range_nrmse(y_true, y_pred):
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
        range_ = np.ptp(y_true, axis=0)
        normalized_rmse = np.max(rmse/range_)
        
        return normalized_rmse
        
    @staticmethod
    def std_nrmse(y_true, y_pred):
        
        # rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
        # std = np.std(y_true, axis=0)
        # normalized_rmse = np.mean(rmse/std)
        
        return np.sqrt(mean_squared_error(y_true, y_pred))/np.std(y_true)
    
    @staticmethod        
    def r2(y_true, y_pred):
        
        r2 = r2_score(y_true, y_pred, multioutput='raw_values')
        return np.mean(r2)
    
    @staticmethod        
    def max_rmse(y_true, y_pred):
        
        max_rmse_ = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
        
        return np.max(max_rmse_)
    
    @staticmethod
    def nmax_ae(y_true, y_pred):
        
        nmaxae = np.max(np.abs(y_true - y_pred), axis=0)/np.max(np.abs(y_true - np.mean(y_true, axis=0)))
        
        return np.mean(nmaxae)
    
    def test_set(self):
        
        test_inputs, test_outputs = self.test_data        
        test_predictions = self.model.predict(test_inputs)
        
        rmse_ = self.rmse(test_outputs, test_predictions)
        range_nrmse_ = self.range_nrmse(test_outputs, test_predictions)
        std_nrmse_ = self.std_nrmse(test_outputs, test_predictions)
        max_rmse_ = self.max_rmse(test_outputs, test_predictions)
        max_range_nrmse_ = self.max_range_nrmse(test_outputs, test_predictions)
        r2_ = self.r2(test_outputs, test_predictions)
        nmax_ae_ = self.nmax_ae(test_outputs, test_predictions)
        mape_ = self.mape(test_outputs, test_predictions)
        
        
        # """for rmse used in TNN paper"""
        paper_rmse = np.mean([np.sqrt(mean_squared_error(test_outputs[i], test_predictions[i])) for i in range(len(test_predictions))])
        paper_rmse_max = np.max([np.sqrt(mean_squared_error(test_outputs[i], test_predictions[i])) for i in range(len(test_predictions))])

        
        return rmse_, range_nrmse_, std_nrmse_, max_rmse_, max_range_nrmse_, r2_, nmax_ae_, mape_, paper_rmse, paper_rmse_max
    
    def forward_model_get_results(self, function_name, sampler):
        
        test_inputs, test_outputs = self.test_data        

        prediction_output = self.model.predict(test_inputs)
        
        r2 = r2_score(test_outputs, prediction_output)
        rmse = np.sqrt(mean_squared_error(test_outputs, prediction_output))
        mape = mean_absolute_percentage_error(test_outputs, prediction_output)
        nmax_ae = np.mean(np.max(np.abs(test_outputs- prediction_output), axis=0)/np.max(np.abs(test_outputs - np.mean(test_outputs, axis=0))))
        
        
        return r2, rmse, mape, nmax_ae
