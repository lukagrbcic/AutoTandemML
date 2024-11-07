import numpy as np
import sys
sys.path.insert(0, '../../InverseBench/src/')
sys.path.insert(1, 'samplers')

from benchmarks import *
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
                 multifidelity=0, verbose=1, forward_metrics=False, save_data=True):
        
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
        self.save_data = save_data

        self.r2 = []
        self.rmse = []
        self.mape = []
        self.nmax_ae = []
      
        self.r2_foward = []
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
                
            if self.sampler != 'model_uncertainty':
                
                X_sampled = samplers(self.sampler, self.max_samples, self.lb, self.ub, self.algorithm).generate_samples()

                y_sampled = self.evaluator.evaluate(X_sampled)
                
            
                run = AutoTNN(self.evaluator, self.lb, self.ub, self.init_size, 
                              self.batch_size, self.max_samples, self.algorithm, 
                              self.test_data, lf_samples=self.multifidelity, sampler=self.sampler,
                              x_init=X_sampled, y_init=y_sampled)
            else:
                
                run = AutoTNN(self.evaluator, self.lb, self.ub, self.init_size, 
                              self.batch_size, self.max_samples, self.algorithm, 
                              self.test_data, lf_samples=self.multifidelity, sampler=self.sampler)
                
            
            run.get_inverse_DNN()
                    
            if self.forward_metrics is not False:
            
                r2_fwd, rmse_fwd, mape_fwd, nmax_ae_fwd = inverse_model_analysis(test_input, test_output, self.function_name).error_metrics_forward()
                
                self.r2_foward.append(r2_fwd)
                self.rmse_forward.append(rmse_fwd)
                self.mape_forward.append(mape_fwd)
                self.nmax_ae_forward.append(nmax_ae_fwd)

           
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
            
            # np.save(dir_+f'/inverseDNN_{self.sampler}_{self.max_samples}_{self.init_size}_{self.batch_size}_{self.n_runs}_r2.npy', self.r2)
            # np.save(dir_+f'/inverseDNN_{self.sampler}_{self.max_samples}_{self.init_size}_{self.batch_size}_{self.n_runs}_rmse.npy', self.rmse)
            # np.save(dir_+f'/inverseDNN_{self.sampler}_{self.max_samples}_{self.init_size}_{self.batch_size}_{self.n_runs}_mape.npy', self.mape)
            # np.save(dir_+f'/inverseDNN_{self.sampler}_{self.max_samples}_{self.init_size}_{self.batch_size}_{self.n_runs}_nmax_ae.npy', self.nmax_ae)
            
            if self.forward_metrics is not False:
                
                results_forward = {'r2': self.r2_forward,
                            'rmse': self.rmse_forward,
                            'mape': self.mape_forward,
                            'nmax_ae': self.nmax_ae_forward,
                            'sampler': self.sampler,
                            'max_samples': self.max_samples,
                            'init_size': self.init_size,
                            'batch_size':self.batch_size,
                            'n_runs': self.n_runs}
                
                np.save(dir_+f'/forwardDNN_{self.sampler}_{self.n_runs}.npy', results_forward)
                
                            
            #     np.save(dir_+f'/forwardDNN_{self.sampler}_{self.max_samples}_{self.init_size}_{self.batch_size}_{self.n_runs}_r2.npy', self.r2_forward)
            #     np.save(dir_+f'/forwardDNN_{self.sampler}_{self.max_samples}_{self.init_size}_{self.batch_size}_{self.n_runs}_rmse.npy', self.rmse_forward)
            #     np.save(dir_+f'/forwardDNN_{self.sampler}_{self.max_samples}_{self.init_size}_{self.batch_size}_{self.n_runs}_mape.npy', self.mape_forward)
            #     np.save(dir_+f'/forwardDNN_{self.sampler}_{self.max_samples}_{self.init_size}_{self.batch_size}_{self.n_runs}_nmax_aer2.npy', self.nmax_ae_forward)

        if self.forward_metrics is not False:
            return results_, results_forward
       
        else:    
            return results_
        
    
            
        
            

        
        
        
    
    
        