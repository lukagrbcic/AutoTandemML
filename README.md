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

To run the code in order to obtain the PyTorch deep neural network files that contain the weights and architecture:

```python
from .auto_tandem import AutoTNN

import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor

from AutoTandemML.run_experiment import experiment_setup
from AutoTandemML.postprocess_tnn import plot_results



```

