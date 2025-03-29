import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


def create_layer(in_dim, out_dim, batch_norm=False, dropout=0.0):
    layers = [nn.Linear(in_dim, out_dim)]
    init.kaiming_uniform_(layers[0].weight, nonlinearity='relu')
    init.zeros_(layers[0].bias)
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_dim))
    layers.append(nn.ReLU())
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Genonet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers, first_fun=lambda x: x, final_fun=torch.tanh,
                 batch_norm=False, dropout=0.0):
        super(Genonet, self).__init__()
        # Initial conversion
        self.first_fun = first_fun

        # Initial layer from n to m dimensions
        self.initial_layer = create_layer(input_dim, hidden_dim, batch_norm, dropout)

        # Intermediate layers (k of them, each from m to m dimensions)
        self.intermediate_layers = nn.ModuleList(
            [create_layer(hidden_dim, hidden_dim, batch_norm, dropout) for _ in range(hidden_layers)]
        )

        # Final layer to 3 dimensions, no batch norm here
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        self.final_fun = final_fun
        init.kaiming_uniform_(self.final_layer.weight, nonlinearity='relu')
        init.zeros_(self.final_layer.bias)

    def forward(self, x):
        # Initial conversion
        x = self.first_fun(x)

        # Pass through the initial layer
        x = self.initial_layer(x)

        # Pass through each intermediate layer with ReLU activation
        for layer in self.intermediate_layers:
            x = layer(x)

        # Pass through the final layer
        x = self.final_layer(x)
        # Learn tanh for the three outputs
        x = self.final_fun(x)
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
