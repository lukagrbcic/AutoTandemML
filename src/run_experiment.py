import numpy as np
import sys
sys.path.insert(0, '../../InverseBench/src/')
sys.path.insert(1, 'samplers')

from benchmarks import *
from check_accuracy import error
from auto_tandem import AutoTNN
from ensemble_regressor import EnsembleRegressor
from inverse_validator import inverse_model_analysis
from generate_samples import samplers

import shutil
import warnings
warnings.filterwarnings("ignore")


class experiment_setup:
    
    def __init__(self, sampler, n_runs, init_size, batch_size, 
                 max_samples, test_data, algorithm, evaluator, lb, ub, function_name='',
                 multifidelity=0, verbose=1, forward_metrics=True, forward_metrics_dnn=False, save_data=True):
        
        self.sampler = sampler
        self.n_runs = n_runs
        self.init_size = init_size
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.test_data = test_data
        self.algorithm = algorithm
        self.evaluator = evaluator
        self.lb = lb
        self.ub = ub
        self.function_name = function_name
        self.multifidelity = multifidelity
        self.verbose = verbose
        self.forward_metrics = forward_metrics
        self.forward_metrics_dnn = forward_metrics_dnn
        self.save_data = save_data

        self.r2 = []
        self.rmse = []
        self.mape = []
        self.nmax_ae = []
      
        self.r2_forward_dnn = []
        self.rmse_forward_dnn = []
        self.mape_forward_dnn = []
        self.nmax_ae_forward_dnn = []
        
        self.r2_forward = []
        self.rmse_forward = []
        self.mape_forward = []
        self.nmax_ae_forward = []
        
    def clear_files(self):
        
        shutil.rmtree('inverseDNN')
        if os.path.exists('inverseDNN'):
            None
        else:
            os.mkdir('inverseDNN')

        if os.path.exists('forwardDNN'):
            os.remove('forwardDNN')
        if os.path.exists('model_config.npy'):
            os.remove('model_config.npy')

    def run(self):
        
        test_input, test_output = self.test_data
        
        for i in range(self.n_runs):
            
            self.clear_files()
            
            if i%self.verbose == 0:
                
                print ('Run', i+1)
                print (self.sampler)
                
            if self.sampler not in ['model_uncertainty', 'ensemble', 'modelHC_uncertainty', 'model_quantile', 'modelLHS_quantile']:
                
                X_sampled = samplers(self.sampler, self.max_samples, self.lb, self.ub, self.algorithm).generate_samples()

                y_sampled = self.evaluator.evaluate(X_sampled)

                run = AutoTNN(self.evaluator, self.lb, self.ub, self.init_size, 
                              self.batch_size, self.max_samples, self.algorithm, 
                              self.test_data, lf_samples=self.multifidelity, sampler=self.sampler, return_forward_data=True,
                              x_init=X_sampled, y_init=y_sampled)
                
                
                model, X_hf, y_hf = run.get_inverse_DNN()
                
                
                if self.forward_metrics is not False:
                    
                    model = self.algorithm[1].fit(X_hf, y_hf)
                    
                    r2_fwd, rmse_fwd, mape_fwd, nmax_ae_fwd = error(model, self.test_data).forward_model_get_results(self.function_name, self.sampler)
                    self.r2_forward.append(r2_fwd)
                    self.rmse_forward.append(rmse_fwd)
                    self.mape_forward.append(mape_fwd)
                    self.nmax_ae_forward.append(nmax_ae_fwd)
                
            else:
                
                run = AutoTNN(self.evaluator, self.lb, self.ub, self.init_size, 
                              self.batch_size, self.max_samples, self.algorithm, 
                              self.test_data, lf_samples=self.multifidelity, sampler=self.sampler, return_forward_data=True)
                
                model, X_hf, y_hf = run.get_inverse_DNN()
                
                
                if self.forward_metrics is not False:
                    
                    r2_fwd, rmse_fwd, mape_fwd, nmax_ae_fwd = error(model, self.test_data).forward_model_get_results(self.function_name, self.sampler)
                    self.r2_forward.append(r2_fwd)
                    self.rmse_forward.append(rmse_fwd)
                    self.mape_forward.append(mape_fwd)
                    self.nmax_ae_forward.append(nmax_ae_fwd)
                
                    
            if self.forward_metrics_dnn is not False:
            
                r2_fwd_dnn, rmse_fwd_dnn, mape_fwd_dnn, nmax_ae_fwd_dnn = inverse_model_analysis(test_input, test_output, self.function_name).error_metrics_forward()
                
                self.r2_forward_dnn.append(r2_fwd_dnn)
                self.rmse_forward_dnn.append(rmse_fwd_dnn)
                self.mape_forward_dnn.append(mape_fwd_dnn)
                self.nmax_ae_forward_dnn.append(nmax_ae_fwd_dnn)

           
            r2, rmse, mape, nmax_ae = inverse_model_analysis(test_input, test_output, self.function_name).error_metrics()
            
            self.r2.append(r2)
            self.rmse.append(rmse)
            self.mape.append(mape)
            self.nmax_ae.append(nmax_ae)
            
            if i%self.verbose == 0:
                print ('mean R2:', np.mean(self.r2), np.std(self.r2))
                print ('mean RMSE:', np.mean(self.rmse), np.std(self.rmse))
                print ('mean MAPE:', np.mean(self.mape), np.std(self.mape))
                print ('mean NMAX_AE:', np.mean(self.nmax_ae), np.std(self.nmax_ae))
            
        if self.verbose > 0:
            print ('final results')
            print ('mean R2:', np.mean(self.r2), np.std(self.r2))
            print ('mean RMSE:', np.mean(self.rmse), np.std(self.rmse))
            print ('mean MAPE:', np.mean(self.mape), np.std(self.mape))
            print ('mean NMAX_AE:', np.mean(self.nmax_ae), np.std(self.nmax_ae))
            print ('====================================================')
        
        if self.save_data == True:
            
            dir_ = f'{self.function_name}_results'
            
            if os.path.exists(dir_):
                None
            else:
                os.mkdir(dir_)
                
            results_ = {'r2': self.r2,
                        'rmse': self.rmse,
                        'mape': self.mape,
                        'nmax_ae': self.nmax_ae,
                        'sampler': self.sampler,
                        'max_samples': self.max_samples,
                        'init_size': self.init_size,
                        'batch_size':self.batch_size,
                        'n_runs': self.n_runs}
            
            np.save(dir_+f'/inverseDNN_{self.sampler}_{self.n_runs}.npy', results_)
            
            
            if self.forward_metrics_dnn is not False:
                
                results_forward_dnn = {'r2': self.r2_forward_dnn,
                            'rmse': self.rmse_forward_dnn,
                            'mape': self.mape_forward_dnn,
                            'nmax_ae': self.nmax_ae_forward_dnn,
                            'sampler': self.sampler,
                            'max_samples': self.max_samples,
                            'init_size': self.init_size,
                            'batch_size':self.batch_size,
                            'n_runs': self.n_runs}
                
                np.save(dir_+f'/forwardDNN_{self.sampler}_{self.n_runs}.npy', results_forward_dnn)
            
                        
            if self.forward_metrics is not False:
                
                results_forward= {'r2': self.r2_forward,
                            'rmse': self.rmse_forward,
                            'mape': self.mape_forward,
                            'nmax_ae': self.nmax_ae_forward,
                            'sampler': self.sampler,
                            'max_samples': self.max_samples,
                            'init_size': self.init_size,
                            'batch_size':self.batch_size,
                            'n_runs': self.n_runs}
                
                np.save(dir_+f'/forward_model_{self.sampler}_{self.n_runs}.npy', results_forward)
                
                            

        if self.forward_metrics_dnn is not False:
            return results_, results_forward_dnn
        
        elif self.forward_metrics is not False:
            return results_, results_forward
        
        else:    
            return results_
        
    
            
        
            

        
        
        
    
    
        