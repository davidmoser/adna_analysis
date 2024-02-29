import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# n: Input dimension
# m: hidden dimension
# k: number of layers
class SimpleGenoNet(nn.Module):
    def __init__(self, n, m, k):
        super(SimpleGenoNet, self).__init__()
        # Initial layer from n to m dimensions
        self.initial_layer = nn.Linear(n, m)
        init.xavier_uniform_(self.initial_layer.weight)
        init.zeros_(self.initial_layer.bias)

        # Intermediate layers (k of them, each from m to m dimensions)
        self.intermediate_layers = nn.ModuleList()
        for _ in range(k):
            layer = nn.Linear(m, m)
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)
            self.intermediate_layers.append(layer)

        # Final layer to 3 dimensions
        self.final_layer = nn.Linear(m, 3)
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

        # Post-processing for output ranges
        # Age: Linear output, may need scaling/clipping to [0, 100000] after prediction
        # Longitude: Scaled tanh output to [-180, 180]
        # Latitude: Scaled tanh output to [-90, 90]
        age, longitude, latitude = x[:, 0], x[:, 1], x[:, 2]
        age = torch.exp((torch.tanh(age) + 1) * 5)
        longitude = 180 * torch.tanh(longitude)
        latitude = 90 * torch.tanh(latitude)

        # Combine and return the processed outputs
        return torch.stack([age, longitude, latitude], dim=1)


# Example usage
n = 10  # Number of input dimensions (genotypes)
m = 20  # Number of dimensions for intermediate layers
k = 5  # Number of intermediate layers

model = SimpleGenoNet(n, m, k)
# Example input tensor (batch size of 1 for demonstration)
input_tensor = torch.randint(0, 3, (1, n)).float()
output = model(input_tensor)

print(output)
