<div align="center">

![Logo](https://github.com/lukagrbcic/AutoTandemML/blob/main/autotandemml.png?raw=true)

</div>
# AutoTandemML
Automated Tandem Neural Networks (TNN) for inverse design problems in science and engineering.

AutoTandemML utilizes active learning methods to efficiently generate a dataset to train a Tandem Neural Network for inverse design challenges. 

## Table of Contents
- [AutoTandemML](#autotandemml)
- [Overview](#overview)
- [Usage](#usage)
- [Results](#results)
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

# Define your function evaluator (implementation dependent)
def function_evaluator(x):
    """
    Code that generates a response based on a design vector x
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
sampler = 'model_uncertainty'

# Set up and run the experiment
run_exp = experiment_setup(sampler, init_size, batch_size, max_samples, 
                           algorithm, function_evaluator, lower_boundary, upper_boundary)
run_exp.run()

# After completion, the inverse DNN files are saved in the inverseDNN folder.


```

## Results
After running the experiments, the trained inverse Deep Neural Network files are saved in the `inverseDNN` folder. Here’s a brief overview of the expected files:
- `.pth`: Model weights
- `model_config.npy`: Model architecture

## License
This project is licensed under the XXX License. See the LICENSE file for details.
