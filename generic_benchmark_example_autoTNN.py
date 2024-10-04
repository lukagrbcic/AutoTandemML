import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from xgboost import XGBRegressor

sys.path.insert(0, '../InverseBench/src/')
sys.path.insert(1, 'src')
sys.path.insert(2, 'src/samplers')

from benchmarks import *
from auto_tandem import AutoTNN
from ensemble_regressor import EnsembleRegressor


import warnings
warnings.filterwarnings("ignore")


bench = 'airfoils'
name = 'airfoil_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)


lb, ub = f.get_bounds()


test_input = np.load(f'../InverseBench/test_data/{bench}_data/input_test_data.npy')
test_output = np.load(f'../InverseBench/test_data/{bench}_data/output_test_data.npy')
test_data = (test_input, test_output)


init_size=5
batch_size=5
max_samples=20
sampler='model_entropy'

ensemble = [XGBRegressor(n_estimators=i[1], reg_lambda=i[0]) for i in [[0.1, 10], [0.5,50], [0.8, 75], [1,100], [10, 125]]]             
algorithm = ('xgb_ensemble', EnsembleRegressor(ensemble))
             
             
run = AutoTNN(f, lb, ub, init_size, batch_size, max_samples, algorithm, test_data)

forward_model = run.get_foward_model()