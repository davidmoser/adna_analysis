import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class SimpleGenoNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers):
        super(SimpleGenoNet, self).__init__()
        # Initial layer from n to m dimensions
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        init.xavier_uniform_(self.initial_layer.weight)
        init.zeros_(self.initial_layer.bias)

        # Intermediate layers (k of them, each from m to m dimensions)
        self.intermediate_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)
            self.intermediate_layers.append(layer)

        # Final layer to 3 dimensions
        self.final_layer = nn.Linear(hidden_dim, 3)
        init.xavier_uniform_(self.final_layer.weight)
        init.zeros_(self.final_layer.bias)

    def forward(self, x):
        # Pass through the initial layer
        x = F.relu(self.initial_layer(x))

        # Pass through each intermediate layer with ReLU activation
        for layer in self.intermediate_layers:
            x = F.relu(layer(x))

        # Pass through the final layer
        x = self.final_layer(x)
        # Learn tanh for the three outputs
        # for conversion to actual age and coordinates see below
        x = torch.tanh(x)
        return x

    # Post-processing for output ranges
    # Age: 1 to 100000
    # Longitude: -180, 180
    # Latitude: -90, 90
    @staticmethod
    def real_to_train(x):
        age, longitude, latitude = x[:, 0], x[:, 1], x[:, 2]
        age = (torch.log(age + 1) / 6 - 1).clamp(max=1)
        longitude = longitude / 180
        latitude = latitude / 90
        return torch.stack([age, longitude, latitude], dim=1)

    @staticmethod
    def real_to_train_single(x):
        age, longitude, latitude = x[0], x[1], x[2]
        age = (np.log(age + 1) / 6 - 1).clip(-1, 1)
        longitude = longitude / 180
        latitude = latitude / 90
        return np.array([age, longitude, latitude])

    @staticmethod
    def train_to_real(x):
        age, longitude, latitude = x[:, 0], x[:, 1], x[:, 2]
        age = torch.exp(6 * (age + 1)) - 1
        longitude = 180 * longitude
        latitude = 90 * latitude
        return torch.stack([age, longitude, latitude], dim=1)
