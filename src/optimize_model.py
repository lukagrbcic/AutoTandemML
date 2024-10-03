from sklearn.datasets import make_regression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from ensemble_regressor import EnsembleRegressor
import pandas as pd

class optimize:
    
    def __init__(self, algorithm, X, y, n_iter=100, cv=3, ensemble_size=5):
        
        self.algorithm = algorithm
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.cv = cv
        self.ensemble_size = ensemble_size
        
    def get_param_dist(self):
        
        
        
        if 'rf' in self.algorithm[0]:
            
            param_dist = {
                'n_estimators': randint(50, 300),            
                'max_depth': randint(1, 30),                 
                'min_samples_split': randint(2, 20),         
                'min_samples_leaf': randint(1, 20),          
                'max_features': [0.5, 0.8, 1],    
                'bootstrap': [True, False]                   
            }
            
        if 'xgb' in self.algorithm[0]:
            
            param_dist = {
                'n_estimators': randint(50, 300),           
                'max_depth': randint(1, 15),                
                'learning_rate': uniform(0.01, 0.3),        
                'subsample': uniform(0.5, 0.5),              
                'colsample_bytree': uniform(0.5, 0.5),       
                'colsample_bylevel': uniform(0.5, 0.5),      
                'min_child_weight': randint(1, 10),          
                'gamma': uniform(0, 0.5),
                'reg_alpha': uniform(0, 1),                  
                'reg_lambda': uniform(0, 1)                 
            }
            
          
        return param_dist
            
    
    def search(self, model):
        
        random_search = RandomizedSearchCV(
            model, 
            param_distributions=self.get_param_dist(), 
            n_iter=self.n_iter, 
            cv=self.cv, 
            verbose=0, 
            n_jobs=-1,
            return_train_score=True
        )

        random_search.fit(self.X, self.y)
        results = pd.DataFrame(random_search.cv_results_)

        sorted_results = results.sort_values(by='mean_test_score', ascending=False)
        
        return sorted_results
        
    
    def get_hyperparameters(self):
        
        
        
        if self.algorithm[0] == 'rf':
            
            self.ensemble_size = 1
            
            sorted_results = self.search(RandomForestRegressor())
            
            top_n_results = sorted_results.head(self.ensemble_size)
            parameters = [i for i in top_n_results['params']]
                  
            
            for i in range(len(parameters)):
                model = RandomForestRegressor(n_estimators=parameters[i]['n_estimators'],
                                          max_depth=parameters[i]['max_depth'],
                                          min_samples_split=parameters[i]['min_samples_split'],
                                          min_samples_leaf=parameters[i]['min_samples_leaf'],
                                          max_features=parameters[i]['max_features'],
                                          bootstrap=parameters[i]['bootstrap'])
        
        elif self.algorithm[0] =='xgb':
            
            self.ensemble_size = self.ensemble_size
            
            sorted_results = self.search(XGBRegressor())
            
            top_n_results = sorted_results.head(self.ensemble_size)
            parameters = [i for i in top_n_results['params']]
            
            
            ensemble = []
            for i in range(len(parameters)):
                ensemble.append(XGBRegressor(n_estimators=parameters[i]['n_estimators'],
                                          max_depth=parameters[i]['max_depth'],
                                          learning_rate=parameters[i]['learning_rate'],
                                          subsample=parameters[i]['subsample'],
                                          colsample_bytree=parameters[i]['colsample_bytree'],
                                          colsample_bylevel=parameters[i]['colsample_bylevel'],
                                          min_child_weight=parameters[i]['min_child_weight'],
                                          gamma=parameters[i]['gamma'],
                                          reg_alpha=parameters[i]['reg_alpha'],
                                          reg_lambda=parameters[i]['reg_lambda']))
                            
            model = EnsembleRegressor(ensemble)
        
            
        return model


        
        
        
        
        
        
        
        
        
        
        
        
        
    