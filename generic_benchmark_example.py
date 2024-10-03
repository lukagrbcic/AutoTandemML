import sys

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sys.path.insert(0, '../InverseBench/src/')
sys.path.insert(1, 'src')
sys.path.insert(2, 'src/samplers')

from benchmarks import *
import active_learning as al
import postprocess as pp

bench = 'airfoils'
name = 'airfoil_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)
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
                             
init_size=10
batch_size=5
max_samples=200
n_repeats=1
sampler='unc'
algorithm = ('rf', RandomForestRegressor(n_estimators=20))

results = []

run = al.activeLearner(f, lb, ub,
                       init_size, batch_size,
                       max_samples, sampler,
                       algorithm,
                       test_data, verbose=batch_size)


file_path = f'./{bench}_results/{sampler}_{max_samples}_{batch_size}_{n_repeats}.npy'

if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        results_exp = np.load(file_path, allow_pickle=True).item()

else:
    print("File does not exist, continuing.")

    results_exp = run.run(n_repeats)
    np.save(f'./{bench}_results/{sampler}_{max_samples}_{batch_size}_{n_repeats}.npy', results_exp)


results.append(results_exp)



sampler='unc_hc'
run = al.activeLearner(f, lb, ub,
                       init_size, batch_size,
                       max_samples, sampler,
                       algorithm,
                       test_data, verbose=batch_size)


file_path = f'./{bench}_results/{sampler}_{max_samples}_{batch_size}_{n_repeats}.npy'

if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        results_exp = np.load(file_path, allow_pickle=True).item()

else:
    print("File does not exist, continuing.")

    results_exp = run.run(n_repeats)
    np.save(f'./{bench}_results/{sampler}_{max_samples}_{batch_size}_{n_repeats}.npy', results_exp)


results.append(results_exp)



sampler='unc_lhs_pso'
run = al.activeLearner(f, lb, ub,
                       init_size, batch_size,
                       max_samples, sampler,
                       algorithm,
                       test_data, verbose=batch_size)


file_path = f'./{bench}_results/{sampler}_{max_samples}_{batch_size}_{n_repeats}.npy'

if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        results_exp = np.load(file_path, allow_pickle=True).item()

else:
    print("File does not exist, continuing.")

    results_exp = run.run(n_repeats)
    np.save(f'./{bench}_results/{sampler}_{max_samples}_{batch_size}_{n_repeats}.npy', results_exp)


results.append(results_exp)


sampler='random'
run = al.activeLearner(f, lb, ub,
                       init_size, batch_size,
                       max_samples, sampler,
                       algorithm,
                       test_data, verbose=batch_size)

file_path = f'./{bench}_results/{sampler}_{max_samples}_{batch_size}_{n_repeats}.npy'

if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        results_rnd = np.load(file_path, allow_pickle=True).item()

else:
    print("File does not exist, continuing.")

    results_rnd = run.run(n_repeats)
    np.save(f'./{bench}_results/{sampler}_{max_samples}_{batch_size}_{n_repeats}.npy', results_exp)
    
results.append(results_rnd)







pp.plot_results(results).compare_metrics()
