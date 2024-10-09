import numpy as np
import sys
from scipy.stats import qmc


sys.path.insert(0, 'samplers')
sys.path.insert(1, 'models')

import check_accuracy as ca
from generate_samples import samplers
import optimize_model as opt
from ensemble_regressor import EnsembleRegressor
import warnings
warnings.filterwarnings("ignore")


class activeLearner:
    
    
    def __init__(self, function, lb, ub, init_size, 
                       batch_size, max_samples, sampler, algorithm, test_data, 
                       var=None,
                       hyperparameters=np.inf, 
                       initial_hyperparameter_search=False,
                       init_samples=[], 
                       verbose=1,
                       return_model=False, return_hf_samples=False):
        
        self.function = function
        self.lb = lb
        self.ub = ub
        self.init_size = init_size
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.sampler = sampler
        self.test_data = test_data
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters
        self.initial_hyperparameter_search = initial_hyperparameter_search
        self.init_samples = init_samples
        self.verbose = verbose
        self.return_model = return_model
        self.return_hf_samples = return_hf_samples
        
    def model_update(self, X, y):
                
        self.model.fit(X,y)

        return self.model
    
    def model_optimization(self, X, y):
        
        self.model = opt.optimize(self.algorithm, X, y).get_hyperparameters()

    def initialize(self):

        X = samplers('lhs', self.init_size, self.lb, self.ub, self.algorithm).generate_samples(11)
        y = self.function.evaluate(X)
        
        return X, y
    
    def get_samples(self, sampled_points=[]):
                
        X = samplers(self.sampler, self.batch_size, 
                     self.lb, self.ub, self.algorithm, sampled_points, self.model).generate_samples()
        
        return X
        
    def loop(self):
        
        X, y = self.initialize()
        
        if self.initial_hyperparameter_search == True:
            if self.verbose > 0:
                print ('Initial hyperparameter search!')
            self.model_optimization(X, y)
        else:
            self.model = self.algorithm[1]
            
            
        self.model = self.model_update(X, y)
        
       # rmse, range_nrmse, std_nrmse, max_rmse, max_range_nrmse, r2, nmax_ae, mape, paper_rmse, paper_rmse_max = ca.error(self.model, self.test_data).test_set()
        rmse, range_nrmse, std_nrmse, max_rmse, max_range_nrmse, r2, nmax_ae, mape = ca.error(self.model, self.test_data).test_set()

    
        size = [len(X)]
        r2_ = [r2]
        mape_ = [mape]
        rmse_ = [rmse]
        range_nrmse_ = [range_nrmse]
        std_nrmse_ = [std_nrmse]
        max_rmse_ = [max_rmse]
        max_range_nrmse_ = [max_range_nrmse]
        nmax_ae_ = [nmax_ae]
        # paper_rmse_ = [paper_rmse]
        # paper_rmse_max_ = [paper_rmse_max]
    
        if self.verbose > 0:        
            print ('RMSE:', rmse, 
                    '\nNRMSE:', range_nrmse,
                    '\nMAX RMSE:', max_rmse, 
                    '\nMAX NRMSE:', max_range_nrmse, 
                    '\nR2:', r2,
                    '\nNMAX AE:',  nmax_ae,
                    '\nMAPE:', mape)
                    # '\nPAPER RMSE:', paper_rmse,
                    # '\nPAPER RMSE MAX', paper_rmse_max)
                
        while len(X) <= self.max_samples - self.init_size:
            
            X_new = self.get_samples(sampled_points=X)
            y_new = self.function.evaluate(X_new)
            
            X = np.vstack((X, X_new))
            y = np.vstack((y, y_new))
                        
            if len(X)%self.hyperparameters == 0:
                print ('Optimizing model!')
                self.model_optimization(X, y)
                
            self.model = self.model_update(X, y)

            
            # rmse, range_nrmse, std_nrmse, max_rmse, max_range_nrmse, r2, nmax_ae, mape, paper_rmse, paper_rmse_max = ca.error(self.model, self.test_data).test_set()
            rmse, range_nrmse, std_nrmse, max_rmse, max_range_nrmse, r2, nmax_ae, mape = ca.error(self.model, self.test_data).test_set()

            if self.verbose > 0 and len(X)%self.verbose == 0:
                print ('-----------------------------------')
                print ('RMSE:', rmse, 
                        '\nNRMSE:', range_nrmse,
                        '\nMAX RMSE:', max_rmse, 
                        '\nMAX NRMSE:', max_range_nrmse, 
                        '\nR2:', r2,
                        '\nNMAX AE:',  nmax_ae,
                        '\nMAPE:', mape)
                        # '\nPAPER RMSE:', paper_rmse,
                        # '\mPAPER RMSE MAX', paper_rmse_max)
                
                print ('Size', len(X))

            size.append(len(X))
            r2_.append(r2)
            mape_.append(mape)
            rmse_.append(rmse)
            range_nrmse_.append(range_nrmse)
            std_nrmse_.append(std_nrmse)
            max_rmse_.append(max_rmse)
            max_range_nrmse_.append(max_range_nrmse)
            nmax_ae_.append(nmax_ae)
            # paper_rmse_.append(paper_rmse)
            # paper_rmse_max_.append(paper_rmse_max)
            
            
        if self.verbose > 0:
            print ('-----------------------------------')
            print ('Run finished!')
            print ('Sampler used:', self.sampler)
            print ('-----------------------------------')
            print ('RMSE:', rmse, 
                    '\nNRMSE:', range_nrmse,
                    '\nMAX RMSE:', max_rmse, 
                    '\nMAX NRMSE:', max_range_nrmse, 
                    '\nR2:', r2,
                    '\nNMAX AE:',  nmax_ae,
                    '\nMAPE:', mape)
                    # '\nPAPER RMSE:', paper_rmse,
                    # '\mPAPER RMSE MAX', paper_rmse_max)
            
            print ('Size', len(X))
        

        return r2_, mape_, rmse_, range_nrmse_, std_nrmse_, max_rmse_, max_range_nrmse_, nmax_ae_, size, self.model, X, y
    
    def run(self, n_repeats=1):
        
        r2_array = []
        mape_array = []
        rmse_array = []
        range_nrmse_array = []
        std_nrmse_array = []
        max_rmse_array = []
        max_range_nrmse_array = []
        nmax_ae_array = []
        size_array = []
        

        for i in range(n_repeats):
            if self.verbose > 0:
                print ('====================')
                print ('Starting run ', i)
                print ('====================')
            
            r2_, mape_, rmse_, range_nrmse_, std_nrmse_, max_rmse_, max_range_nrmse_, nmax_ae_, size, self.model, X, y = self.loop()
            
            r2_array.append(r2_)
            mape_array.append(mape_)
            rmse_array.append(rmse_)
            range_nrmse_array.append(range_nrmse_)
            std_nrmse_array.append(std_nrmse_)
            max_rmse_array.append(max_rmse_)
            max_range_nrmse_array.append(max_range_nrmse_)
            nmax_ae_array.append(nmax_ae_)
            size_array.append(size)
        
        r2_array = np.array(r2_array)
        mape_array = np.array(mape_array)
        rmse_array = np.array(rmse_array)
        range_nrmse_array = np.array(range_nrmse_array)
        std_nrmse_array = np.array(std_nrmse_array)
        max_rmse_array = np.array(max_rmse_array)
        max_range_nrmse_array = np.array(max_range_nrmse_array)
        nmax_ae_array = np.array(nmax_ae_array)
        size_array = np.array(size_array)
        
        
        results = { 'r2_array': r2_array,
                    'mape_array': mape_array,
                    'rmse_array': rmse_array,
                    'range_nrmse_array': range_nrmse_array,
                    'std_nrmse_array': std_nrmse_array,
                    'max_rmse_array': max_rmse_array,
                    'max_range_nrmse_array': max_range_nrmse_array,
                    'nmax_ae_array': nmax_ae_array,
                    'size': size_array[0],
                    'mode':f'{self.sampler}_{self.algorithm[0]}',
                    'n_runs': n_repeats,
                    'batch_size': self.batch_size}
        
        if self.return_model == True and self.return_hf_samples == True:
            return results, self.model, X, y 
        else:
            return results
            
        
        
        
        
        
        
        
        
        
        
        
        
