import sys

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
import random

sys.path.insert(0, '../InverseBench/src/')
sys.path.insert(1, 'src')
sys.path.insert(2, 'src/samplers')

from benchmarks import *
from ensemble_regressor import EnsembleRegressor
from run_experiment import experiment_setup
from postprocess_tnn import plot_results

import warnings
warnings.filterwarnings("ignore")



bench = 'scalar_diffusion' 
name = 'scalar_diffusion_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)

# ensemble_size = 5
# n_est = np.arange(10, 210, ensemble_size)
# reg_lambda = np.linspace(0.1, 10, ensemble_size)
# list_ = [[reg_lambda[i], n_est[i]] for i in range(ensemble_size)]
# ensemble = [XGBRegressor(n_estimators=i[1], reg_lambda=i[0]) for i in list_]             
# algorithm = ('xgb', EnsembleRegressor(ensemble))

ensemble = []
for i in range(10):
    ensemble.append(make_pipeline(MinMaxScaler(), MLPRegressor(hidden_layer_sizes=(100, 200, 100), 
                                                                  random_state=random.randint(10, 250))))
algorithm = ('mlp_ensemble', EnsembleRegressor(ensemble))      


lb, ub = f.get_bounds()

test_input = np.load(f'../InverseBench/test_data/{bench}_data/input_test_data.npy')[:1000]
test_output = np.load(f'../InverseBench/test_data/{bench}_data/output_test_data.npy')[:1000]
test_data = (test_input, test_output)

all_results = []

init_size=20
batch_size=5
max_samples=200
n_runs = 10
# sampler = 'random'

# scalar_setup = experiment_setup(sampler, n_runs, init_size, batch_size, max_samples, 
#                                 test_data, algorithm, f, lb, ub, function_name=name)

# file_path = f'./{name}_results/inverseDNN_{sampler}_{n_runs}.npy'

# if os.path.exists(file_path):
#     with open(file_path, 'r') as file:
#         results = np.load(file_path, allow_pickle=True).item()
# else:
#     results = scalar_setup.run()

# all_results.append(results)

# sampler = 'lhs'

# scalar_setup = experiment_setup(sampler, n_runs, init_size, batch_size, max_samples, 
#                                 test_data, algorithm, f, lb, ub, function_name=name)

# file_path = f'./{name}_results/inverseDNN_{sampler}_{n_runs}.npy'

# if os.path.exists(file_path):
#     with open(file_path, 'r') as file:
#         results = np.load(file_path, allow_pickle=True).item()
# else:
#     results = scalar_setup.run()
# all_results.append(results)

sampler = 'model_uncertainty'

scalar_setup = experiment_setup(sampler, n_runs, init_size, batch_size, max_samples, 
                                test_data, algorithm, f, lb, ub, function_name=name)

file_path = f'./{name}_results/inverseDNN_{sampler}_{n_runs}.npy'

if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        results = np.load(file_path, allow_pickle=True).item()
else:
    results = scalar_setup.run()
all_results.append(results)


plot_results(all_results).compare_metrics()


