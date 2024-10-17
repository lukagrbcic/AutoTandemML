import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=None,
                 dropout=0.0, batch_norm=False, activation='relu', output_activation=None):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_size
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_size))
        
        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
 
        self.network = nn.Sequential(*layers)
        self.device = device
        self.to(device)  

    def forward(self, x):
        return self.network(x)


class ModelFactory:
    def __init__(self):
        self.models = {
            'mlp': MLP
            # 'cnn': CNN,
        }

    def create_model(self, model_type, **kwargs):
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not recognized.")
        return self.models[model_type](**kwargs)


