import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.0, batch_norm=False, activation='relu'):
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
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CNN(nn.Module):
    def __init__(self, input_channels, conv_layers, fc_layers, output_size, dropout=0.0, batch_norm=False, pooling='max', activation='relu'):
        super(CNN, self).__init__()
        layers = []
        in_channels = input_channels
        for out_channels, kernel_size, stride, padding in conv_layers:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            if pooling == 'max':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif pooling == 'avg':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        
        self.flatten = nn.Flatten()
        in_dim = self._get_conv_output(input_channels, conv_layers)
        fc_layers = [in_dim] + fc_layers + [output_size]
        fc = []
        for i in range(len(fc_layers) - 1):
            fc.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))
            if i < len(fc_layers) - 2:
                if batch_norm:
                    fc.append(nn.BatchNorm1d(fc_layers[i + 1]))
                if activation == 'relu':
                    fc.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    fc.append(nn.LeakyReLU())
                elif activation == 'tanh':
                    fc.append(nn.Tanh())
                if dropout > 0:
                    fc.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*fc)

    def _get_conv_output(self, input_channels, conv_layers):
        o = torch.zeros(1, input_channels, 32, 32)  # assuming input size 32x32
        o = self.conv(o)
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class ModelFactory:
    def __init__(self):
        self.models = {
            'mlp': MLP,
            'cnn': CNN,
        }

    def create_model(self, model_type, **kwargs):
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not recognized.")
        return self.models[model_type](**kwargs)

# Usage
factory = ModelFactory()

# MLP example
mlp_model = factory.create_model('mlp', input_size=100, hidden_layers=[128, 64], output_size=10, dropout=0.2, batch_norm=True, activation='leaky_relu')
print(mlp_model)

# CNN example
cnn_model = factory.create_model('cnn', input_channels=3, conv_layers=[(16, 3, 1, 1), (32, 3, 1, 1)], fc_layers=[128, 64], output_size=10, dropout=0.2, batch_norm=True, pooling='avg', activation='tanh')
print(cnn_model)

