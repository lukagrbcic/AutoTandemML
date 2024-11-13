import numpy as np





name = 'inconel_benchmark'
all_results_inverse = []

n_runs = 30

sampler = 'random'

file_path_inverse = f'./{name}_results/inverseDNN_{sampler}_{n_runs}.npy'
results_inverse = np.load(file_path_inverse, allow_pickle=True).item()
all_results_inverse.append(results_inverse)
sampler = 'lhs'

file_path_inverse = f'./{name}_results/inverseDNN_{sampler}_{n_runs}.npy'
results_inverse = np.load(file_path_inverse, allow_pickle=True).item()
all_results_inverse.append(results_inverse)
sampler = 'model_uncertainty'

file_path_inverse = f'./{name}_results/inverseDNN_{sampler}_{n_runs}.npy'
results_inverse = np.load(file_path_inverse, allow_pickle=True).item()


all_results_inverse.append(results_inverse)

sampler = 'bc'

file_path_inverse = f'./{name}_results/inverseDNN_{sampler}_{n_runs}.npy'
results_inverse = np.load(file_path_inverse, allow_pickle=True).item()


all_results_inverse.append(results_inverse)

sampler = 'greedyfp'

file_path_inverse = f'./{name}_results/inverseDNN_{sampler}_{n_runs}.npy'
results_inverse = np.load(file_path_inverse, allow_pickle=True).item()


all_results_inverse.append(results_inverse)

metric_keys = ['rmse', 'r2', 'nmax_ae']
# Method names
methods = ['TNN$_R$', 'TNN$_{LHS}$', 'TNN$_{AL}$', 'TNN$_{BC}$', 'TNN$_{GFP}$']

# Map metric keys to display names
metric_display_names = {
    'rmse': 'RMSE',
    'r2': 'R$^2$',
    # 'mape': 'MAPE',
    'nmax_ae': 'NMAE'
}

# Initialize a dictionary to store statistics
statistics = {}

# Compute statistics for each method and metric
for i, method in enumerate(methods):
    statistics[method] = {}
    method_data = all_results_inverse[i]
    for metric_key in metric_keys:
        metric_data = method_data[metric_key]
        mean = np.mean(metric_data)
        std = np.std(metric_data)
        max_val = np.max(metric_data)
        min_val = np.min(metric_data)
        statistics[method][metric_key] = {
            'mean': mean,
            'std': std,
            'max': max_val,
            'min': min_val
        }

# Generate LaTeX table
latex_table = r'''
\begin{table}[ht]
\centering
\caption{Statistical Summary of Metrics for Different Methods}
\begin{tabular}{lccccc}
\hline
\textbf{Metric} & \textbf{Method} & \textbf{Mean} & \textbf{Std} & \textbf{Max} & \textbf{Min} \\
\hline
'''

for metric_key in metric_keys:
    metric_name = metric_display_names[metric_key]
    for method in methods:
        stat = statistics[method][metric_key]
        latex_table += f"{metric_name} & {method} & " \
                       f"{stat['mean']:.4f} & {stat['std']:.4f} & " \
                       f"{stat['max']:.4f} & {stat['min']:.4f} \\\\\n"

latex_table += r'''\hline
\end{tabular}
\end{table}
'''

# Output the LaTeX table
print(latex_table)