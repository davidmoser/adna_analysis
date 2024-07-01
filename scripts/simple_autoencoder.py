import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def create_layer(in_dim, out_dim, batch_norm=True):
    layers = [nn.Linear(in_dim, out_dim)]
    init.kaiming_uniform_(layers[0].weight, nonlinearity='relu')
    init.zeros_(layers[0].bias)
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_dim))
    return nn.Sequential(*layers)

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_layers, latent_dim):
        super(SimpleAutoencoder, self).__init__()
        # Encoder
        # Initial layer from 'input_dim' to 'hidden_dim' dimensions
        self.encoder_initial = create_layer(input_dim, hidden_dim)

        # Encoder stack ('hidden_layers' layers, each from 'hidden_dim' to 'hidden_dim' dimensions)
        self.encoder_stack = nn.ModuleList([create_layer(hidden_dim, hidden_dim) for _ in range(hidden_layers)])

        # To latent space
        self.encoder_latent = create_layer(hidden_dim, latent_dim, batch_norm=False)  # No batch norm in the latent layer

        # Decoder
        # To hidden dim
        self.decoder_latent = create_layer(latent_dim, hidden_dim)

        # Decoder stack
        self.decoder_stack = nn.ModuleList([create_layer(hidden_dim, hidden_dim) for _ in range(hidden_layers)])

        # Final layer from 'hidden_dim' to 'input_dim' dimensions
        self.decoder_final = create_layer(hidden_dim, input_dim, batch_norm=False)  # No batch norm in the final layer

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Pass through the encoder initial layer
        x = F.relu(self.encoder_initial(x))

        # Pass through each layer in the encoder stack with ReLU activation
        for layer in self.encoder_stack:
            x = F.relu(layer(x))

        # Pass through the encoder latent layer
        x = self.encoder_latent(x)
        return x

    def decode(self, x):
        # Pass through the decoder latent layer
        x = F.relu(self.decoder_latent(x))

        # Pass through each layer in the decoder stack with ReLU activation
        for layer in self.decoder_stack:
            x = F.relu(layer(x))

        # Pass through the decoder final layer
        x = self.decoder_final(x)
        # No final activation, we output logits
        return x
