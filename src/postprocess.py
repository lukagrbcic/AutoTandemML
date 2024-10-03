import matplotlib.pyplot as plt
import numpy as np

from matplotlib.markers import MarkerStyle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import os
import sys

plt.rcParams.update({
    "text.usetex": True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{sfmath} \sffamily \usepackage{upgreek}',
    "font.size": 18,
})

# class postprocess_dict:
    
#     def __init__(self, results_):
        
#         self.results_ = results_
    
#     def get_dict(self):
        
#         full_res_dict = {'evals': None, 'fitness': [] , 'n_runs': None,
#                          'optimizer': None, 'solutions': [] }

#         full_res_dict['evals'] = self.results_[0]['evals']
#         full_res_dict['n_runs'] = np.sum([res['n_runs'] for res in self.results_])
#         full_res_dict['optimizer'] = self.results_[0]['optimizer']
#         full_res_dict['fitness'] = np.concatenate([res['fitness'] for res in self.results_])
#         full_res_dict['solutions'] = np.concatenate([res['solutions'] for res in self.results_])

#         return full_res_dict
    


                    
class plot_results:
    
    def __init__(self, results):
        
        self.results = results
    
        
    def compare_metrics(self, std_show=True):
            
        colors = ['red', 'blue', 'orange', 'green', 'cyan', 'black', 'purple', 'gold', 'pink',
                  'magenta', 'lime', 'gray', 'maroon', 'navy', 'olive', 'teal', 'aqua', 'silver', 'chocolate' ]

        
        n_runs = self.results[0]['n_runs']
        size = len(self.results[0]['size'])
        batch_size = self.results[0]['batch_size']
        
        metrics = ['r2', 'mape', 'rmse', 'range_nrmse', 'std_nrmse', 'max_rmse', 'max_range_nrmse', 'nmax_ae']
        
        
        for m in metrics:
            plt.figure(figsize=(6,5))
            c = 0
            for dict_ in self.results:
                
                    size = dict_['size']
                    array_ = dict_[f'{m}_array']
    
                    mean_array = np.mean(array_, axis=0)
                    
                    if std_show == True:
                        std_array = np.std(array_, axis=0)#[:len(fitness[0])]#*100
                        min_array = np.min(array_, axis=0)
                        max_array = np.max(array_, axis=0)
                        perc10 = np.percentile(array_, 10, axis=0)
                        perc90 = np.percentile(array_, 90, axis=0)
    
                                   
                    plt.plot(size, mean_array, color=colors[c],alpha=1, linewidth=2, marker='o', label = f"{dict_['mode']}")
                    
                    if std_show == True:
                        plt.fill_between(size, mean_array, perc10, color=colors[c],alpha=0.025)
                        plt.fill_between(size, mean_array, perc90, color=colors[c],alpha=0.025)
                        plt.plot(size, perc10, linewidth=0.5, color=colors[c],alpha=1)
                        plt.plot(size, perc90, linewidth=0.5, color=colors[c],alpha=1)
                        
          
    
                    plt.xlabel('Samples')
                    if m == 'r2': metric = 'R$^2$' 
                    else: metric = m.upper()
                    plt.ylabel(f'{metric}')
                    plt.legend(fontsize=12)
                    c+=1
                    
                    ax = plt.gca()
    
                    for axis in ['top', 'bottom', 'left', 'right']:
                      ax.spines[axis].set_linewidth(2)
                    
                    plt.tight_layout()
                    plt.title(f"batch size: {batch_size}, batches: {len(size)}")
                
                # plt.savefig(f'results/mean_rmse_{n_runs}_convergence_rate_std_{std_show}.pdf', dpi=400)   

        

        

            
        
        
        