import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.simple_autoencoder import SimpleAutoencoder
from scripts.simple_geno_net import SimpleGenoNet

# Instantiate models
snp_dim = 9190
hidden_dim, hidden_layers = 100, 20
inverse_geno_net = SimpleGenoNet(3, snp_dim, hidden_dim, hidden_layers)
simple_autoencoder = SimpleAutoencoder(snp_dim, hidden_dim, hidden_layers, 3)

# Load saved weights
inverse_geno_net.load_state_dict(torch.load('../models/inverse_geno_net.pth'))
simple_autoencoder.load_state_dict(torch.load('../models/simple_autoencoder.pth'))

# Set models to evaluation mode
inverse_geno_net.eval()
simple_autoencoder.eval()

# Define ranges for location and time
latitude_range = np.linspace(-90, 90, 18)
longitude_range = np.linspace(0, 360, 36)
age_range = np.linspace(10000, 1000, 10)

# Generate frames
fig = plt.figure()
ims = []

for age in age_range:
    print(f"Creating image for age {age}")
    # Create a batch for the current age with all latitude and longitude combinations
    latitudes, longitudes = np.meshgrid(latitude_range, longitude_range)
    age_locations = np.vstack([np.full(latitudes.size, age), latitudes.flatten(), longitudes.flatten()]).T
    age_locations_tensor = torch.tensor(age_locations, dtype=torch.float32)

    # Predict SNPs for the batch
    predicted_snps = inverse_geno_net(SimpleGenoNet.real_to_train(age_locations_tensor))
    decoded_colors = simple_autoencoder.encode(predicted_snps).detach().numpy()

    # Normalize the colors
    normalized_colors = (decoded_colors + np.array([10, 10, 15])) / np.array([45, 20, 20])

    # Create frames for each latitude and longitude
    for normalized_color in normalized_colors:
        im = plt.imshow(np.full((10, 10, 3), normalized_color), animated=True)
        ims.append([im])

# Create the movie
print("Creating the movie")
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)

# Save the movie
print("Saving the movie")
ani.save('../results/geno_movie.gif', writer='pillow')

plt.show()
