# AutoTandemML
Automated Tandem Neural Networks (TNN) for inverse design problems in science and engineering.

AutoTandemML utilizes active learning methods to efficiently generate a dataset to train a Tandem Neural Network for inverse design challenges. 

-----------------

The process consists of three main segments:

1. Sampling: Generating a deteset with active learning 

2. TNN: Training the forward Deep Neural Network (x -> f(x)) 

3. TNN: Training the inverse Deep Neural Network (f(x) -> x)

-----------------

Currently, the only active learning algorithms that are supported are Random Forests (scikit-learn implementation), and a more
general ensemble of models that is defined as a list of scikit-learn models.

Sampling algorithms that are currently supported are active learning, random sampling, latin hypercube sampling, greedyFP sampling and 
best candidate sampling.

-----------------

To run the code in order to obtain the PyTorch Deep Neural Network files that contain the weights and architecture:

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from AutoTandemML.run_experiment import experiment_setup #we import the main run wrapper


def function_evaulator(x): #this is the evaluator function used for active learning
    """
    Code that generates a response based on a design vector x
    """
    return response

#we must define the lower and upper boundary of our design space vector x
lower_boundary = ...
upper_boundary = ...


#we use the RF algorithm for the active learning process
algorithm = ('rf', RandomForestRegressor()) #we define the RF algorithm as a tuple, the first value should be 'rf' if we want hyperparameter optimization

init_size=20 #the initial sample size we generate with latin hypercube sampling 
batch_size=5 #the batch size for active learning (i.e. how many new samples we generate per each iteration)
max_samples=150 #the maximum number of samples we want to generate for the TNN training


sampler = 'model_uncertainty' #this is the basic active learning approach where we use the uncertainty to find new points


# this is the object we define for a single run (active learning + forward and inverse DNN) 
run_exp = experiment_setup(sampler, init_size, batch_size, max_samples, 
                                algorithm, function_evaulator, lower_boundary, upper_boundary)

#run the experimental setup
run_exp.run()


```
After completion, the inverse Deep Neural Network files (.pth and model_config.npy) are saved in the inverseDNN folder.