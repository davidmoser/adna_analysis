import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.basemap import Basemap, maskoceans

from scripts.simple_autoencoder import SimpleAutoencoder
from scripts.simple_geno_net import SimpleGenoNet

# Instantiate models
snp_dim = 9190
hidden_dim, hidden_layers = 150, 10
inverse_geno_net = SimpleGenoNet(3, snp_dim * 4, hidden_dim, hidden_layers, final_fun=lambda x: x)
simple_autoencoder = SimpleAutoencoder(snp_dim * 4, hidden_dim, hidden_layers, 3)

# Load saved weights
inverse_geno_net.load_state_dict(torch.load('../models/inverse_geno_net.pth'))
simple_autoencoder.load_state_dict(torch.load('../models/simple_autoencoder.pth'))

# Set models to evaluation mode
inverse_geno_net.eval()
simple_autoencoder.eval()

# Define ranges for location and time
latitude_min, latitude_max = 0, 70
longitude_min, longitude_max = -20, 150
latitude_range = np.linspace(latitude_min, latitude_max, 180)
longitude_range = np.linspace(longitude_min, longitude_max, 360)
age_range = np.linspace(10000, 1000, 10)

# Generate and display images
for age in age_range:
    print(f"Creating image for age {age}")
    # Create a batch for the current age with all latitude and longitude combinations
    latitudes, longitudes = np.meshgrid(latitude_range, longitude_range)
    age_locations = np.vstack([np.full(latitudes.size, age), longitudes.flatten(), latitudes.flatten()]).T
    age_locations_tensor = torch.tensor(age_locations, dtype=torch.float32)

    # Predict SNPs for the batch
    predicted_snps = inverse_geno_net(SimpleGenoNet.real_to_train(age_locations_tensor))
    colors = simple_autoencoder.encode(predicted_snps).detach().numpy()

    # Normalize the colors
    normalized_colors = (colors + np.array([-100, 660, -200])) / np.array([350, 550, 650])

    # Create an image where each point has the color corresponding to its latitude and longitude
    img = np.zeros((latitude_range.size, longitude_range.size, 3))

    for i in range(len(normalized_colors)):
        lon_idx = i // latitude_range.size
        lat_idx = i % latitude_range.size
        img[lat_idx, lon_idx, :] = normalized_colors[i]

    # Initialize Basemap
    plt.figure(figsize=(10, 10 * (latitude_max - latitude_min) // (longitude_max - longitude_min)))
    m = Basemap(projection='cyl', llcrnrlat=latitude_min, urcrnrlat=latitude_max, llcrnrlon=longitude_min,
                urcrnrlon=longitude_max, resolution='c')
    m.drawcoastlines()
    m.drawcountries()

    # Create a land-sea mask
    x, y = np.meshgrid(longitude_range, latitude_range)
    mask = maskoceans(x, y, img[:, :, 0], inlands=False)

    # Apply the land-sea mask to set ocean pixels to white
    img[mask.mask] = [1, 1, 1]  # RGB for white

    # Plot the image with longitude and latitude
    m.imshow(img, interpolation='nearest', origin='lower')

    plt.title(f"Age: {age}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    plt.clf()
    plt.close()
    del age_locations_tensor, predicted_snps, colors, normalized_colors, img  # Free up memory
    torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
