import sys

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sys.path.insert(0, '../InverseBench/src/')
sys.path.insert(1, 'src')
sys.path.insert(2, 'src/samplers')


from benchmarks import *
import active_learning as al

bench = 'friedman'
name = 'friedman_multioutput_benchmark'
# model = load_model(name).load()
f = benchmark_functions(name)#), model)
lb, ub = f.get_bounds()


test_input = np.load(f'../InverseBench/test_data/{bench}_data/input_test_data.npy')
test_output = np.load(f'../InverseBench/test_data/{bench}_data/output_test_data.npy')
test_data = (test_input, test_output)



# function = lambda x: f.evaluate(x)


# airfoil_inverse_model = AutoTNN(evaluation_function,
#                              lb, ub,
#                              max_evals, 
#                              init_size, 
                             # test_data)
                             
init_size=5
batch_size=10
max_samples=750
sampler='unc_lhs_pso'
algorithm = ('rf', RandomForestRegressor(n_estimators=68))

run = al.activeLearner(f, lb, ub,
                       init_size, batch_size,
                       max_samples, sampler,
                       algorithm,
                       test_data, verbose=10)

run.run()