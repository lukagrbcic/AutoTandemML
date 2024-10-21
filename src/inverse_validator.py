import numpy as np
from sklearn.metrics import mean_squared_error

"""
1) X_test, y_test
2) y_test -> inverse model -> X_prediction -> forward model -> y_prediction 
3) compare NEPD(X_pred, X_test) and error(y_pred, y_test)
""" 


class inverse_model_analysis:
    def __init__(self, X_test, y_test, forward_model, inverse_model):
        
        