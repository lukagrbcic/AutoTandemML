import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import random

sys.path.insert(0, '../InverseBench/src/')
sys.path.insert(1, 'src')
sys.path.insert(2, 'src/samplers')

from benchmarks import *
import active_learning as al
import postprocess as pp
from ensemble_regressor import EnsembleRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

import warnings
warnings.filterwarnings("ignore")


bench = 'friedman' #(deep ensembles)
name = 'friedman_multioutput_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)

# bench = 'inconel' #(random forests)
# name = 'inconel_benchmark'
# model = load_model(name).load()
# f = benchmark_functions(name, model)

# bench = 'airfoils' #(xgb ensembles)
# name = 'airfoil_benchmark'
# model = load_model(name).load()
# f = benchmark_functions(name, model)



bench = 'scalar_diffusion' #(deep ensembles)
name = 'scalar_diffusion_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)
algorithm = ('rf', RandomForestRegressor())

lb, ub = f.get_bounds()
test_input = np.load(f'../InverseBench/test_data/{bench}_data/input_test_data.npy')
test_output = np.load(f'../InverseBench/test_data/{bench}_data/output_test_data.npy')
test_data = (test_input, test_output)

init_size=20
batch_size=20
max_samples=200
n_repeats=1
sampler='model_uncertainty'
# np.random.seed(43)

# algorithm = ('rf', RandomForestRegressor())

# ensemble_size = 5
# n_est = np.arange(10, 210, ensemble_size)
# reg_lambda = np.linspace(0.1, 10, ensemble_size)
# list_ = [[reg_lambda[i], n_est[i]] for i in range(ensemble_size)]
# ensemble = [XGBRegressor(n_estimators=i[1], reg_lambda=i[0]) for i in list_]             
          
# algorithm = ('xgb_ensemble', EnsembleRegressor(ensemble))
      

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

ensemble = []
for i in range(50):
    # ensemble.append(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 200, 100), 
    #                                                               random_state=random.randint(10, 250))))
    ensemble.append(make_pipeline(MinMaxScaler(), MLPRegressor(hidden_layer_sizes=(100, 200, 100), 
                                                                  random_state=random.randint(10, 250))))
algorithm = ('mlp_ensemble', EnsembleRegressor(ensemble))           

results = []
run = al.activeLearner(f, lb, ub,
                        init_size, batch_size,
                        max_samples, sampler,
                        algorithm,
                        test_data)#, initial_hyperparameter_search=True)


file_path = f'./{bench}_results/{sampler}_{max_samples}_{batch_size}_{n_repeats}_{algorithm[0]}.npy'

if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        results_exp2 = np.load(file_path, allow_pickle=True).item()

else:
    print("File does not exist, continuing.")

    results_exp2 = run.run(n_repeats)
    np.save(file_path, results_exp2)


results.append(results_exp2)


sampler='random'
run = al.activeLearner(f, lb, ub,
                        init_size, batch_size,
                        max_samples, sampler,
                        algorithm,
                        test_data)#, initial_hyperparameter_search=True)

file_path = f'./{bench}_results/{sampler}_{max_samples}_{batch_size}_{n_repeats}_{algorithm[0]}.npy'

if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        results_rnd = np.load(file_path, allow_pickle=True).item()

else:
    print("File does not exist, continuing.")

    results_rnd = run.run(n_repeats)
    np.save(file_path, results_rnd)

results.append(results_rnd)

pp.plot_results(results).compare_metrics(fname=bench)
