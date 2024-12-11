<div align="center">

![Logo](https://github.com/lukagrbcic/AutoTandemML/blob/main/autotandemml.png?raw=true)

</div>

## AutoTandemML
Automated Tandem Neural Networks (TNN) for inverse design problems in science and engineering.

AutoTandemML utilizes active learning methods to efficiently generate a dataset to train a Tandem Neural Network for inverse design challenges. 

## Table of Contents
- [AutoTandemML](#autotandemml)
- [Overview](#overview)
- [Usage](#usage)
- [Samplers](#samplers)
- [Results](#results)
- [References](#references)
- [License](#license)

-----------------
## Overview

The process consists of three main segments:

1. Sampling: Generating a dateset (x, f(x)) with active learning

2. TNN: Training the forward Deep Neural Network (x -> f(x)) 

3. TNN: Training the inverse Deep Neural Network (f(x) -> x)

-----------------

## Usage
Here is a basic example of how to use AutoTandemML to train your models:

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from AutoTandemML.run_experiment import experiment_setup

# Define your function evaluator
def function_evaluator(x):
    """
    Code that generates a response based on a design vector x, 
    e.g. a CFD simulation of a flow around an airfoil (response), 
    based on the geometry of the airfoil (x)
    """
    return response

# Set design space boundaries
lower_boundary = ...
upper_boundary = ...

# Initialize the active learning algorithm
algorithm = ('rf', RandomForestRegressor())
init_size = 20  # Initial sample size
batch_size = 5   # Batch size for active learning
max_samples = 150 # Maximum samples for training

# Define the sampler
sampler = 'model_uncertainty' #model_uncertainty is the basic active learning sampler

# Set up and run the experiment
run_exp = experiment_setup(sampler, init_size, batch_size, max_samples, 
                           algorithm, function_evaluator, lower_boundary, upper_boundary)
run_exp.run()

# After completion, the inverse DNN files are saved in the inverseDNN folder.


```

By default, the forward and inverse DNNs are optimized with random search (10 iterations).
If we want to set the number of random search evaluations, we define the combinations parameter:

```python

#combinations=100 means 100 evaluations of MLP hyperparameters

experiment_setup(sampler, init_size, batch_size, max_samples, 
                 algorithm, function_evaulator, lb, ub, function_name=name, combinations=100).run()
                 
```

Here is an example of how to use AutoTandemML to train your models using the EnsembleRegressor class:

```python
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

#here we define an ensemble of 10 deep neural networks
ensemble = []
for i in range(10):
    ensemble.append(make_pipeline(MinMaxScaler(), MLPRegressor(hidden_layer_sizes=(100, 200, 100), 
                                                                  random_state=random.randint(10, 250))))
algorithm = ('mlp_ensemble', EnsembleRegressor(ensemble))      


lb, ub = f.get_bounds()


all_results = []

init_size=20
batch_size=5
max_samples=400
n_runs = 30


sampler = 'greedyfp' #in this case we use the greedyFP sampler (not active learning)

scalar_setup = experiment_setup(sampler, init_size, batch_size, max_samples, 
                                algorithm, function_evaulator, lb, ub, function_name=name).run()


```
## Samplers
Samplers that can be used to generate the dataset for TNN training are:
- `random`: Random sampling
- `lhs`: Latin Hypercube sampling
- `greedyFP`: greedyFP sampling [1]
- `bc`: best candidate sampling [1]
- `model_uncertainty`: active learning sampling (based on prediction uncertainty)
- `model_entropy`: active learning sampling (based on prediction entropy)
- `model_quantile`: active learning sampling (based on quantile uncertainty)
- `ensemble`: active learning sampling that combines a batch of the uncertainty, entropy and quantile samplers


Some experimental samplers 



## Results
After running the experiments, the trained inverse Deep Neural Network files are saved in the `inverseDNN` folder. Here’s a brief overview of the expected files:
- `.pth`: Model weights
- `model_config.npy`: Model architecture

The sampled dataset used for training the inverese Deep Neural Network is also saved:
- `X_hf.npy`: x (inverse DNN outputs)
- `y_hf.npy`: f(x) (inverse DNN inputs)

## References
- `[1]`: Kamath, C. (2022). Intelligent sampling for surrogate modeling, hyperparameter optimization, and data analysis. 
         Machine Learning with Applications, 9, 100373.


## License
This project is licensed under the XXX License. See the LICENSE file for details.
