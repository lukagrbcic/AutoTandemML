import sys
import numpy as np


sys.path.insert(0, 'src')
sys.path.insert(1, '../InverseBench/src/')


from optimize_inverse import get_hyperparameters


param_dist = {
    'model_type': ['mlp'],
    'hidden_layers': [[64], [128], [256], [128, 128],
                      [256, 256], [512, 512], [64, 128, 64],
                      [128, 256, 128], [256, 512, 256], [64, 128, 256, 128, 64]],
    'dropout': [0.0, 0.2],
    'batch_norm': [False, True],
    'activation': ['relu', 'leaky_relu'],
    'epochs': [100, 200, 300, 1000],
    'batch_size': [32, 64],
    'learning_rate': [0.001, 0.01, 0.1],
    'input_scaler': [None, 'MinMax', 'Standard'],
    'output_scaler': [None, 'MinMax', 'Standard'],
    'output_activation': [None]
}

bench = 'airfoils'
test_input = np.load(f'../InverseBench/test_data/{bench}_data/input_test_data.npy')
test_output = np.load(f'../InverseBench/test_data/{bench}_data/output_test_data.npy')

optimize = get_hyperparameters(test_input, test_output, param_dist, n_iter=2).run()
