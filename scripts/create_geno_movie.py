import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.basemap import Basemap, maskoceans

from scripts.autoencoder import Autoencoder
from scripts.genonet import Genonet

# Instantiate models
dims = 4932052
hidden_dim, hidden_layers = 150, 10
inverse_geno_net = Genonet(3, dims, hidden_dim, hidden_layers, final_fun=lambda x: x, batch_norm=True)
autoencoder = Autoencoder(dims, hidden_dim, hidden_layers, 3)

# Load saved weights
inverse_geno_net.load_state_dict(torch.load('./inverse_genonet.pth'))
autoencoder.load_state_dict(torch.load('./autoencoder.pth'))

# Set models to evaluation mode
inverse_geno_net.eval()
autoencoder.eval()
torch.set_grad_enabled(False)

# Define ranges for location and time
# Europe
latitude_min, latitude_max = 30, 70
longitude_min, longitude_max = -15, 45
# Europe, Asia, Africa
# latitude_min, latitude_max = -40, 85
# longitude_min, longitude_max = -20, 150
latitude_range = np.linspace(latitude_min, latitude_max, 180)
longitude_range = np.linspace(longitude_min, longitude_max, 360)
age_range = np.logspace(start=4, stop=2, num=200, base=10, dtype=np.int64)
color_age_range = np.logspace(start=4, stop=2, num=10, base=10, dtype=np.int64)

# Batch size for processing
batch_size = 64

# Create a list to store filenames
filenames = []

color_min = np.array([1e5, 1e5, 1e5])
color_max = np.array(-color_min)

# First find range of colors for normalization
for age in color_age_range:
    print(f"Creating colors for age {age}")
    # Create a batch for the current age with all latitude and longitude combinations
    latitudes, longitudes = np.meshgrid(latitude_range, longitude_range)
    age_locations = np.vstack([np.full(latitudes.size, age), longitudes.flatten(), latitudes.flatten()]).T
    age_locations_tensor = torch.tensor(age_locations, dtype=torch.float32)

    # Process in batches
    color_batches = []
    num_samples = age_locations_tensor.size(0)
    for i in range(0, num_samples, batch_size):
        batch = age_locations_tensor[i:i+batch_size]
        # Preprocess input as required by the network
        batch_input = Genonet.real_to_train(batch)
        predicted_snps = inverse_geno_net(batch_input)
        # For normalization, use one_hot=False
        batch_colors = autoencoder.encode(predicted_snps, one_hot=False)
        color_batches.append(batch_colors.cpu().detach())
    colors = torch.cat(color_batches, dim=0).numpy()

    # Normalize the colors
    color_min = np.minimum(color_min, np.min(colors, axis=0))
    color_max = np.maximum(color_max, np.max(colors, axis=0))

    del age_locations_tensor, color_batches, colors  # Free up memory
    torch.cuda.empty_cache()  # Clear GPU memory if using CUDA

# Generate and display images
for age in age_range:
    filename = f"./movie/image_{age}.png"
    filenames.append(filename)

    if os.path.exists(filename):
        continue

    print(f"Creating image for age {age}")
    # Create a batch for the current age with all latitude and longitude combinations
    latitudes, longitudes = np.meshgrid(latitude_range, longitude_range)
    age_locations = np.vstack([np.full(latitudes.size, age), longitudes.flatten(), latitudes.flatten()]).T
    age_locations_tensor = torch.tensor(age_locations, dtype=torch.float32)

    # Process in batches
    color_batches = []
    num_samples = age_locations_tensor.size(0)
    for i in range(0, num_samples, batch_size):
        batch = age_locations_tensor[i:i+batch_size]
        batch_input = Genonet.real_to_train(batch)
        predicted_snps = inverse_geno_net(batch_input)
        # For image generation, use the default one_hot setting
        batch_colors = autoencoder.encode(predicted_snps, one_hot=False)
        color_batches.append(batch_colors.detach())
    colors = torch.cat(color_batches, dim=0).cpu().numpy()

    # Normalize the colors
    normalized_colors = (colors - color_min) / (color_max - color_min)

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

    # Save the image to a file
    plt.savefig(filename)

    plt.clf()
    plt.close()
    del age_locations_tensor, color_batches, colors, normalized_colors, img  # Free up memory
    torch.cuda.empty_cache()  # Clear GPU memory if using CUDA

# Create a movie from the saved images using OpenCV
frame = cv2.imread(filenames[0])
height, width, layers = frame.shape

video = cv2.VideoWriter(filename='./geno_movie.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=5,
                        frameSize=(width, height))

for filename in filenames:
    video.write(cv2.imread(filename))
    os.remove(filename)  # Remove the file after adding to the video

# cv2.destroyAllWindows()
video.release()

torch.set_grad_enabled(True)
