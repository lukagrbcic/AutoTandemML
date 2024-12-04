import matplotlib.pyplot as plt
import numpy as np

from matplotlib.markers import MarkerStyle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import os
import sys
from collections import defaultdict

plt.rcParams.update({
    "text.usetex": True,
    'font.family': 'sans-serif',
    'text.latex.preamble': r'\usepackage{sfmath} \sffamily \usepackage{upgreek}',
    "font.size": 18,
})


class plot_results:

    def __init__(self, results):
        self.results = results

    def compare_metrics(self, path, modelname):
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import defaultdict

        colors = ['red', 'blue', 'orange', 'green']

        metrics = ['r2', 'rmse', 'nmax_ae']

        # Mapping from sampler to method names
        # sampler_to_method = {
        #     'random': 'TNN$_R$',
        #     'lhs': 'TNN$_{LHS}$',
        #     'model_uncertainty': 'TNN$_{AL}$',
        #     'bc': 'TNN$_{BC}$',
        #     'greedyfp': 'TNN$_{GFP}$'
        # }
        
        if modelname == 'inverse':
            sampler_to_method = {
                'random': '$\mathbf{I_{DNN_R}}$',
                'lhs': '$\mathbf{I_{DNN_{LHS}}}$',
                'model_uncertainty': '$\mathbf{I_{DNN_{AL}}}$',
                'bc': '$\mathbf{I_{DNN_{BC}}}$',
                'greedyfp': '$\mathbf{I_{DNN_{GFP}}}$'
            }
        elif modelname == 'forward':
            sampler_to_method = {
                'random': '$\mathbf{M_{R}}$',
                'lhs': '$\mathbf{M_{LHS}}$',
                'model_uncertainty': '$\mathbf{M_{AL}}$',
                'bc': '$\mathbf{M_{BC}}$',
                'greedyfp': '$\mathbf{M_{GFP}}$'
            }


        method_data = defaultdict(lambda: defaultdict(list))

        # Collect data
        for dict_ in self.results:
            sampler = dict_.get('sampler')
            method_name = sampler_to_method.get(sampler, sampler)

            for m in metrics:
                array_ = dict_.get(m)
                if array_ is not None:
                    array_ = np.asarray(array_).ravel() 
                    method_data[method_name][m].extend(array_)
        
        methods = [sampler_to_method['random'], sampler_to_method['lhs'],
                   sampler_to_method['model_uncertainty'], sampler_to_method['bc'], sampler_to_method['greedyfp']]
        
        
        method_colors = {
            sampler_to_method['random']: 'red',
            sampler_to_method['lhs']: 'blue',
            sampler_to_method['model_uncertainty']: 'orange',
            sampler_to_method['bc']: 'green',
            sampler_to_method['greedyfp']: 'purple'
            
        }

        for m in metrics:
            data_to_plot = []
            labels = []
            box_colors = []
            for method in methods:
                if m in method_data[method]:
                    data = method_data[method][m]
                    data_to_plot.append(data)
                    labels.append(method)
                    box_colors.append(method_colors[method])

            if not data_to_plot:
                continue 

            plt.figure(figsize=(6, 5))

            bp = plt.boxplot(data_to_plot, patch_artist=True, labels=labels)

            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)

            if m == 'r2':
                metric_label = 'R$^2$'
                plt.ylim(0.9, 1)
                # plt.yticks(np.arange(0.8, 1+0.04, 0.04))
            elif m == 'nmax_ae':
                metric_label = 'NMAE'
            # elif m == 'nmax_ae':

            else:
                metric_label = m.upper()
                
            # if m == 'mape':
            #     # plt.ylim(0, 0.4)
            
            # if m == 'rmse':
            #     # plt.ylim(0, 0.2)


            plt.ylabel(f'{metric_label}')
            # plt.xlabel('Methods')
            # plt.title(f'Comparison of {metric_label} Across Methods')


            ax = plt.gca()
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2)

            plt.tight_layout()


            plt.savefig(f'{path}_results/{modelname}_{m}_boxplot.pdf', dpi=300)
        

            
        
        
        