from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
import random
import os
import numpy as np


from AutoTandemML.ensemble_regressor import EnsembleRegressor
from AutoTandemML.run_experiment import experiment_setup
from AutoTandemML.postprocess_tnn import plot_results


from InverseBench.benchmarks import load_model, load_test_data, benchmark_functions


bench = 'scalar_diffusion' 
name = 'scalar_diffusion_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)

def function_evaulator(x):
    return f.evaluate(x)


ensemble = []
for i in range(10):
    ensemble.append(make_pipeline(MinMaxScaler(), MLPRegressor(hidden_layer_sizes=(100, 200, 100), 
                                                                  random_state=random.randint(10, 250))))
algorithm = ('mlp_ensemble', EnsembleRegressor(ensemble))      


all_results_inverse = []
all_results_forward = []

lb, ub = f.get_bounds()

test_input, test_output = load_test_data(name).load()
test_data = (test_input[:1000], test_output[:1000])

all_results = []

init_size=20
batch_size=5
max_samples=400
n_runs = 30


sampler = 'random'

scalar_setup = experiment_setup(sampler, n_runs, init_size, batch_size, max_samples, 
                                test_data, algorithm, function_evaulator, lb, ub, function_name=name)
    

file_path_inverse = f'./{name}_results/inverseDNN_{sampler}_{n_runs}.npy'
file_path_forward = f'./{name}_results/forward_model_{sampler}_{n_runs}.npy'

if os.path.exists(file_path_inverse):
    with open(file_path_inverse, 'r') as file:
        results_inverse = np.load(file_path_inverse, allow_pickle=True).item()
    with open(file_path_forward, 'r') as file:
        results_forward = np.load(file_path_forward, allow_pickle=True).item()

else:
    results_inverse, results_forward = scalar_setup.run()

all_results_inverse.append(results_inverse)
all_results_forward.append(results_forward)

sampler = 'lhs'

scalar_setup = experiment_setup(sampler, n_runs, init_size, batch_size, max_samples, 
                                test_data, algorithm, function_evaulator, lb, ub, function_name=name)
    
file_path_inverse = f'./{name}_results/inverseDNN_{sampler}_{n_runs}.npy'
file_path_forward = f'./{name}_results/forward_model_{sampler}_{n_runs}.npy'

if os.path.exists(file_path_inverse):
    with open(file_path_inverse, 'r') as file:
        results_inverse = np.load(file_path_inverse, allow_pickle=True).item()
    with open(file_path_forward, 'r') as file:
        results_forward = np.load(file_path_forward, allow_pickle=True).item()

else:
    results_inverse, results_forward = scalar_setup.run()

all_results_inverse.append(results_inverse)
all_results_forward.append(results_forward)

sampler = 'bc'

scalar_setup = experiment_setup(sampler, n_runs, init_size, batch_size, max_samples, 
                                test_data, algorithm, function_evaulator, lb, ub, function_name=name)
    
file_path_inverse = f'./{name}_results/inverseDNN_{sampler}_{n_runs}.npy'
file_path_forward = f'./{name}_results/forward_model_{sampler}_{n_runs}.npy'

if os.path.exists(file_path_inverse):
    with open(file_path_inverse, 'r') as file:
        results_inverse = np.load(file_path_inverse, allow_pickle=True).item()
    with open(file_path_forward, 'r') as file:
        results_forward = np.load(file_path_forward, allow_pickle=True).item()

else:
    results_inverse, results_forward = scalar_setup.run()

all_results_inverse.append(results_inverse)
all_results_forward.append(results_forward)

sampler = 'greedyfp'

scalar_setup = experiment_setup(sampler, n_runs, init_size, batch_size, max_samples, 
                                test_data, algorithm, function_evaulator, lb, ub, function_name=name)
    
file_path_inverse = f'./{name}_results/inverseDNN_{sampler}_{n_runs}.npy'
file_path_forward = f'./{name}_results/forward_model_{sampler}_{n_runs}.npy'

if os.path.exists(file_path_inverse):
    with open(file_path_inverse, 'r') as file:
        results_inverse = np.load(file_path_inverse, allow_pickle=True).item()
    with open(file_path_forward, 'r') as file:
        results_forward = np.load(file_path_forward, allow_pickle=True).item()

else:
    results_inverse, results_forward = scalar_setup.run()

all_results_inverse.append(results_inverse)
all_results_forward.append(results_forward)

sampler = 'model_uncertainty'

scalar_setup = experiment_setup(sampler, n_runs, init_size, batch_size, max_samples, 
                                test_data, algorithm, function_evaulator, lb, ub, function_name=name)

file_path_inverse = f'./{name}_results/inverseDNN_{sampler}_{n_runs}.npy'
file_path_forward = f'./{name}_results/forward_model_{sampler}_{n_runs}.npy'

if os.path.exists(file_path_inverse):
    with open(file_path_inverse, 'r') as file:
        results_inverse = np.load(file_path_inverse, allow_pickle=True).item()
    with open(file_path_forward, 'r') as file:
        results_forward = np.load(file_path_forward, allow_pickle=True).item()

else:
    results_inverse, results_forward = scalar_setup.run()

all_results_inverse.append(results_inverse)
all_results_forward.append(results_forward)

# plot_results(all_results_inverse).compare_metrics(name,'inverse')
plot_results(all_results_forward).compare_metrics(name,'forward')
