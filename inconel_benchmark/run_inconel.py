import sys
import os
import numpy as np

from sklearn.ensemble import RandomForestRegressor

sys.path.insert(1, 'src')
sys.path.insert(2, 'src/samplers')

from AutoTandemML.run_experiment import experiment_setup
from AutoTandemML.postprocess_tnn import plot_results
from InverseBench.benchmarks import load_model, load_test_data, benchmark_functions

import warnings
warnings.filterwarnings("ignore")


bench = 'inconel'
name = 'inconel_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)

def function_evaulator(x):
    return f.evaluate(x)

algorithm = ('rf', RandomForestRegressor())

lb, ub = f.get_bounds()


all_results_inverse = []
all_results_forward = []

init_size=20
batch_size=5
max_samples=300
n_runs = 1


sampler = 'model_uncertainty'

experiment_setup(sampler, n_runs, init_size, batch_size, max_samples, 
                                test_data, algorithm, function_evaulator, lb, ub, function_name=name).run()


