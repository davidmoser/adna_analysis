import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.basemap import Basemap, maskoceans

from scripts.autoencoder import Autoencoder
from scripts.genonet import Genonet

# Instantiate models
snp_dim = 9190
hidden_dim, hidden_layers = 150, 10
inverse_geno_net = Genonet(3, snp_dim * 4, hidden_dim, hidden_layers, final_fun=lambda x: x)
autoencoder = Autoencoder(snp_dim * 4, hidden_dim, hidden_layers, 3)

# Load saved weights
inverse_geno_net.load_state_dict(torch.load('../models/inverse_genonet_v2.pth'))
autoencoder.load_state_dict(torch.load('../models/autoencoder_v2.pth'))

# Set models to evaluation mode
inverse_geno_net.eval()
autoencoder.eval()

# Define ranges for location and time
# Europe
# latitude_min, latitude_max = 30, 70
# longitude_min, longitude_max = -15, 45
# Europe, Asia, Africa
latitude_min, latitude_max = -40, 85
longitude_min, longitude_max = -20, 150
latitude_range = np.linspace(latitude_min, latitude_max, 180)
longitude_range = np.linspace(longitude_min, longitude_max, 360)
age_range = np.logspace(start=4, stop=2, num=200, base=10, dtype=np.int64)

# Create a list to store filenames
filenames = []

# Generate and display images
for age in age_range:
    print(f"Creating image for age {age}")
    # Create a batch for the current age with all latitude and longitude combinations
    latitudes, longitudes = np.meshgrid(latitude_range, longitude_range)
    age_locations = np.vstack([np.full(latitudes.size, age), longitudes.flatten(), latitudes.flatten()]).T
    age_locations_tensor = torch.tensor(age_locations, dtype=torch.float32)

    # Predict SNPs for the batch
    predicted_snps = inverse_geno_net(Genonet.real_to_train(age_locations_tensor))
    colors = autoencoder.encode(predicted_snps).detach().numpy()

    # Normalize the colors
    color_min = np.min(colors, axis=0)
    color_diff = np.max(colors, axis=0) - color_min
    normalized_colors = (colors - color_min) / color_diff

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
    filename = f"../results/image_{age}.png"
    plt.savefig(filename)
    filenames.append(filename)

    plt.clf()
    plt.close()
    del age_locations_tensor, predicted_snps, colors, normalized_colors, img  # Free up memory
    torch.cuda.empty_cache()  # Clear GPU memory if using CUDA

# Create a movie from the saved images using OpenCV
frame = cv2.imread(filenames[0])
height, width, layers = frame.shape

video = cv2.VideoWriter(filename='../results/geno_movie.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=10,
                        frameSize=(width, height))

for filename in filenames:
    video.write(cv2.imread(filename))
    os.remove(filename)  # Remove the file after adding to the video

cv2.destroyAllWindows()
video.release()
