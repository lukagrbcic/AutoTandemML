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


all_results = []

init_size=20
batch_size=5
max_samples=400
n_runs = 30


sampler = 'greedyfp'

scalar_setup = experiment_setup(sampler, init_size, batch_size, max_samples, 
                                algorithm, function_evaulator, lb, ub, function_name=name).run()
    

