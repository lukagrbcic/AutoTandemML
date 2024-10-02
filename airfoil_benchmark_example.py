import sys

sys.path.insert(0, '../InverseBench/src/')

from benchmarks import *

name = 'airfoil_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)
lb, ub = f.get_bounds()

def evaluation_function(x):
    
    value = f.evaluate(x)

    return value








airfoil_inverse_model = AutoTNN(evaluation_function,
                             lb, ub,
                             max_evals, 
                             init_size, 
                             test_data)