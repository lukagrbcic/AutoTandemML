from sklearn.datasets import make_regression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.neural_network import MLPRegressor
from ensemble_regressor import EnsembleRegressor


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class optimize:
    
    def __init__(self, algorithm, X, y, n_iter=500, cv=3, ensemble_size=50):
        
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
                # 'min_samples_split': randint(2, 20),         
                # 'min_samples_leaf': randint(1, 20),          
                # 'max_features': [0.5, 0.8, 1],    
                # 'bootstrap': [True, False]                   
            }
            
        if 'xgb' in self.algorithm[0]:
            
            param_dist = {
                'n_estimators': randint(50, 300),           
                'max_depth': randint(1, 15),                
                # 'learning_rate': uniform(0.01, 0.3),        
                # 'subsample': uniform(0.5, 0.5),              
                # 'colsample_bytree': uniform(0.5, 0.5),       
                # 'colsample_bylevel': uniform(0.5, 0.5),      
                # 'min_child_weight': randint(1, 10),          
                # 'gamma': uniform(0, 0.5),
                # 'reg_alpha': uniform(0, 1),                  
                # 'reg_lambda': uniform(0, 1)                 
            }
            
        if 'mlp_ensemble' in self.algorithm[0]:
            
            param_dist = {
                'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler(), None],
                'mlp__hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 25), (100, 50, 25)],
                'mlp__activation': ['tanh', 'relu'],
                'mlp__solver': ['sgd', 'adam'],
                'mlp__alpha': [0.0001, 0.001, 0.01],
                'mlp__learning_rate': ['constant', 'adaptive'],
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
            scoring='neg_mean_squared_error',
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
                                          max_depth=parameters[i]['max_depth'])
                                          # min_samples_split=parameters[i]['min_samples_split'],
                                          # min_samples_leaf=parameters[i]['min_samples_leaf'],
                                          # max_features=parameters[i]['max_features'],
                                          # bootstrap=parameters[i]['bootstrap'])
        
        elif self.algorithm[0] =='xgb':
            
            # self.ensemble_size = self.ensemble_size
            
            sorted_results = self.search(XGBRegressor())
            
            top_n_results = sorted_results.head(2*self.ensemble_size)

            
            parameters_list = [i for i in top_n_results['params']]
      
            
            
            parameters = np.random.choice(parameters_list, size=self.ensemble_size, replace=False)
            ensemble = []
            for i in range(len(parameters)):
                ensemble.append(XGBRegressor(n_estimators=parameters[i]['n_estimators'],
                                          max_depth=parameters[i]['max_depth']))
                                          # learning_rate=parameters[i]['learning_rate'],
                                          # subsample=parameters[i]['subsample'],
                                          # colsample_bytree=parameters[i]['colsample_bytree'],
                                          # colsample_bylevel=parameters[i]['colsample_bylevel'],
                                          # min_child_weight=parameters[i]['min_child_weight'],
                                          # gamma=parameters[i]['gamma'],
                                          # reg_alpha=parameters[i]['reg_alpha'],
                                          # reg_lambda=parameters[i]['reg_lambda']))
                            
            model = EnsembleRegressor(ensemble)
            
        elif self.algorithm[0] == 'mlp_ensemble':
            
   
            initial_model = pipeline = Pipeline([
                    ('scaler', StandardScaler()),  # Placeholder, will be overridden by param_dist
                    ('mlp', MLPRegressor())
                ])
            
            sorted_results = self.search(initial_model)
            
            top_n_results = sorted_results.head(self.ensemble_size)
                        
            parameters_list = [i for i in top_n_results['params']]
            parameters = parameters_list
            # parameters = np.random.choice(parameters_list, size=self.ensemble_size, replace=False)
            ensemble = []
            for i in range(len(parameters)):
                ensemble.append(make_pipeline(parameters[i]['scaler'], 
                                              MLPRegressor(hidden_layer_sizes=parameters[i]['mlp__hidden_layer_sizes'],
                                                           activation=parameters[i]['mlp__activation'],
                                                           solver=parameters[i]['mlp__solver'],
                                                           alpha=parameters[i]['mlp__alpha'],
                                                           learning_rate=parameters[i]['mlp__learning_rate'])))
            
            
         
            
            model = EnsembleRegressor(ensemble)
            
        return model


        
        
        
        
        
        
        
        
        
        
        
        
        
    