import sys
import numpy as np


sys.path.append(0, 'src')



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

material = 'airfoil_Re_3_6'
X_train = np.load(f'../data/{material}_data/input_train_data.npy')#[:3000]
y_train = np.load(f'../data/{material}_data/output_train_data.npy')#[:3000]
