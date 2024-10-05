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


bench = 'scalar_diffusion'
name = 'scalar_diffusion_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)
# f = benchmark_functions(name)

lb, ub = f.get_bounds()

test_input = np.load(f'../InverseBench/test_data/{bench}_data/input_test_data.npy')
test_output = np.load(f'../InverseBench/test_data/{bench}_data/output_test_data.npy')
test_data = (test_input, test_output)
                             
init_size=10
batch_size=10
max_samples=200
n_repeats=1
sampler='model_entropy'
# sampler='modelLHS_quantile'
# sampler='modelHC_entropy'

algorithm = ('rf', RandomForestRegressor())

# ensemble = [XGBRegressor(n_estimators=i[1], reg_lambda=i[0]) for i in [[0.1, 10], [0.5,50], [0.8, 75], [1,100], [10, 125]]]             
# ensemble = []
# for i in range(3):
#     ensemble.append(XGBRegressor(n_estimators=random.randint(10, 250), reg_lambda=np.random.uniform(0.01, 10)))


# algorithm = ('xgb_ensemble_3', EnsembleRegressor(ensemble))           

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler



ensemble = []

# for i in range(50):
#     ensemble.append(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 200, 100), 
#                                                                  random_state=random.randint(10, 250))))

# for i in range(50):
#     ensemble.append(make_pipeline(MinMaxScaler(), MLPRegressor(hidden_layer_sizes=(50, 100, 50, 20), 
#                                                                  random_state=random.randint(10, 250))))

# algorithm = ('mlp_ensemble_50', EnsembleRegressor(ensemble))           


# ensemble = []
# for i in range(20):
#     ensemble.append(RandomForestRegressor(n_estimators=random.randint(10, 150)))#, reg_lambda=np.random.uniform(0.01, 10)))


# algorithm = ('rf_ensemble_20', EnsembleRegressor(ensemble))           
           
# ensemble = [RandomForestRegressor(n_estimators=i) for i in [50, 80, 100, 250]]             
# algorithm = ('rfensemble', EnsembleRegressor(ensemble))

results = []
run = al.activeLearner(f, lb, ub,
                        init_size, batch_size,
                        max_samples, sampler,
                        algorithm,
                        test_data, verbose=1)


file_path = f'./{bench}_results/{sampler}_{max_samples}_{batch_size}_{n_repeats}_{algorithm[0]}.npy'

if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        results_exp = np.load(file_path, allow_pickle=True).item()

else:
    print("File does not exist, continuing.")

    results_exp = run.run(n_repeats)
    np.save(file_path, results_exp)


results.append(results_exp)


sampler='random'
run = al.activeLearner(f, lb, ub,
                        init_size, batch_size,
                        max_samples, sampler,
                        algorithm,
                        test_data, verbose=batch_size)

file_path = f'./{bench}_results/{sampler}_{max_samples}_{batch_size}_{n_repeats}_{algorithm[0]}.npy'

if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        results_rnd = np.load(file_path, allow_pickle=True).item()

else:
    print("File does not exist, continuing.")

    results_rnd = run.run(n_repeats)
    np.save(file_path, results_rnd)

results.append(results_rnd)

pp.plot_results(results).compare_metrics()
