import numpy as np





name = 'scalar_diffusion_benchmark'
all_results_inverse = []

n_runs = 30

sampler = 'random'

model_type = 'forward_model'
# model_type = 'inverseDNN'

file_path_inverse = f'./{name}_results/{model_type}_{sampler}_{n_runs}.npy'
results_inverse = np.load(file_path_inverse, allow_pickle=True).item()
all_results_inverse.append(results_inverse)
sampler = 'lhs'

file_path_inverse = f'./{name}_results/{model_type}_{sampler}_{n_runs}.npy'
results_inverse = np.load(file_path_inverse, allow_pickle=True).item()
all_results_inverse.append(results_inverse)
sampler = 'model_uncertainty'

file_path_inverse = f'./{name}_results/{model_type}_{sampler}_{n_runs}.npy'
results_inverse = np.load(file_path_inverse, allow_pickle=True).item()


all_results_inverse.append(results_inverse)

sampler = 'bc'

file_path_inverse = f'./{name}_results/{model_type}_{sampler}_{n_runs}.npy'
results_inverse = np.load(file_path_inverse, allow_pickle=True).item()


all_results_inverse.append(results_inverse)

sampler = 'greedyfp'

file_path_inverse = f'./{name}_results/{model_type}_{sampler}_{n_runs}.npy'
results_inverse = np.load(file_path_inverse, allow_pickle=True).item()


all_results_inverse.append(results_inverse)
metric_keys = ['rmse', 'r2', 'nmax_ae']
# Method names
methods = ['TNN$_R$', 'TNN$_{LHS}$', 'TNN$_{AL}$', 'TNN$_{BC}$', 'TNN$_{GFP}$']

# Map metric keys to display names
metric_display_names = {
    'rmse': 'RMSE',
    'r2': 'R$^2$',
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

# Now, compute the best values for each metric
best_values = {}
for metric_key in metric_keys:
    # For each measure (mean, max, min), find the best value and methods achieving it
    means = {method: statistics[method][metric_key]['mean'] for method in methods}
    maxs = {method: statistics[method][metric_key]['max'] for method in methods}
    mins = {method: statistics[method][metric_key]['min'] for method in methods}

    if metric_key in ['rmse', 'nmax_ae']:  # Lower is better
        best_mean_val = min(means.values())
        best_max_val = min(maxs.values())
        best_min_val = min(mins.values())
    elif metric_key == 'r2':  # Higher is better
        best_mean_val = max(means.values())
        best_max_val = max(maxs.values())
        best_min_val = max(mins.values())
    else:
        raise ValueError(f"Unknown metric {metric_key}")

    # Record best values and methods achieving them
    best_values[metric_key] = {
        'mean': {
            'value': best_mean_val,
            'methods': [method for method, val in means.items() if np.isclose(val, best_mean_val, atol=1e-6)]
        },
        'max': {
            'value': best_max_val,
            'methods': [method for method, val in maxs.items() if np.isclose(val, best_max_val, atol=1e-6)]
        },
        'min': {
            'value': best_min_val,
            'methods': [method for method, val in mins.items() if np.isclose(val, best_min_val, atol=1e-6)]
        },
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
        mean = stat['mean']
        std = stat['std']
        max_val = stat['max']
        min_val = stat['min']

        # Check if values are best and need bolding
        bold_mean = method in best_values[metric_key]['mean']['methods']
        bold_max = method in best_values[metric_key]['max']['methods']
        bold_min = method in best_values[metric_key]['min']['methods']

        mean_fmt = f"{mean:.4f}"
        max_fmt = f"{max_val:.4f}"
        min_fmt = f"{min_val:.4f}"

        if bold_mean:
            mean_fmt = r"\textbf{" + mean_fmt + "}"
        if bold_max:
            max_fmt = r"\textbf{" + max_fmt + "}"
        if bold_min:
            min_fmt = r"\textbf{" + min_fmt + "}"

        latex_table += f"{metric_name} & {method} & " \
                       f"{mean_fmt} & {std:.4f} & " \
                       f"{max_fmt} & {min_fmt} \\\\\n"
    latex_table += "\\hline\n"

latex_table += r'''\end{tabular}
\end{table}
'''

print(latex_table)
# metric_keys = ['rmse', 'r2', 'nmax_ae']
# # Method names
# methods = ['TNN$_R$', 'TNN$_{LHS}$', 'TNN$_{AL}$', 'TNN$_{BC}$', 'TNN$_{GFP}$']

# # Map metric keys to display names
# metric_display_names = {
#     'rmse': 'RMSE',
#     'r2': 'R$^2$',
#     # 'mape': 'MAPE',
#     'nmax_ae': 'NMAE'
# }

# # Initialize a dictionary to store statistics
# statistics = {}

# # Compute statistics for each method and metric
# for i, method in enumerate(methods):
#     statistics[method] = {}
#     method_data = all_results_inverse[i]
#     for metric_key in metric_keys:
#         metric_data = method_data[metric_key]
#         mean = np.mean(metric_data)
#         std = np.std(metric_data)
#         max_val = np.max(metric_data)
#         min_val = np.min(metric_data)
#         statistics[method][metric_key] = {
#             'mean': mean,
#             'std': std,
#             'max': max_val,
#             'min': min_val
#         }

# # Generate LaTeX table
# latex_table = r'''
# \begin{table}[ht]
# \centering
# \caption{Statistical Summary of Metrics for Different Methods}
# \begin{tabular}{lccccc}
# \hline
# \textbf{Metric} & \textbf{Method} & \textbf{Mean} & \textbf{Std} & \textbf{Max} & \textbf{Min} \\
# \hline
# '''

# for metric_key in metric_keys:
#     metric_name = metric_display_names[metric_key]
#     for method in methods:
#         stat = statistics[method][metric_key]
#         latex_table += f"{metric_name} & {method} & " \
#                        f"{stat['mean']:.4f} & {stat['std']:.4f} & " \
#                        f"{stat['max']:.4f} & {stat['min']:.4f} \\\\\n"

# latex_table += r'''\hline
# \end{tabular}
# \end{table}
# '''

# # Output the LaTeX table
# print(latex_table)