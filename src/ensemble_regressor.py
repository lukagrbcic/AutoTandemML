from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class EnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y):
        self.estimators_ = []
        for model in self.models:
            fitted_model = model.fit(X, y)
            self.estimators_.append(fitted_model)
        return self
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.estimators_])
        avg_predictions = np.mean(predictions, axis=0)
        return avg_predictions


