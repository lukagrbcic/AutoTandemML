import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import *
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, '../InverseBench/src/')
sys.path.insert(1, 'src')
sys.path.insert(2, 'src/samplers')

from benchmarks import *
from auto_tandem import AutoTNN
from ensemble_regressor import EnsembleRegressor
from inverse_validator import inverse_model_analysis


import warnings
warnings.filterwarnings("ignore")


# bench = 'airfoils' #(xgb ensemble)
# name = 'airfoil_benchmark'
# model = load_model(name).load()
# f = benchmark_functions(name, model)
# ensemble = [XGBRegressor(n_estimators=i[1], reg_lambda=i[0]) for i in [[0.1, 10], [0.5,50], [0.8, 75], [1,100], [10, 125]]]             
# algorithm = ('xgb_ensemble', EnsembleRegressor(ensemble))

# bench = 'friedman' #(deep ensembles)
# name = 'friedman_multioutput_benchmark'
# model = load_model(name).load()
# f = benchmark_functions(name, model)
# ensemble = []
# for i in range(20):
#     ensemble.append(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 200, 100), 
#                                                                   random_state=i)))
# algorithm = ('mlp_ensemble', EnsembleRegressor(ensemble)) 


bench = 'inconel' #(random forests)
name = 'inconel_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)
algorithm = ('rf', RandomForestRegressor())

lb, ub = f.get_bounds()


test_input = np.load(f'../InverseBench/test_data/{bench}_data/input_test_data.npy')
test_output = np.load(f'../InverseBench/test_data/{bench}_data/output_test_data.npy')
test_data = (test_input, test_output)

init_size=20
batch_size=10
max_samples=200
sampler='model_uncertainty'
# sampler='random'
     
for i in range(10):
    print ('Run', i+1)
    run = AutoTNN(f, lb, ub, init_size, batch_size, max_samples, algorithm, test_data, sampler=sampler, combinations=30)
    run.get_inverse_DNN()
    inverse_model_analysis(test_input, test_output, name, sampler).error_metrics()












