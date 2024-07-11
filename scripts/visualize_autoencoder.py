import matplotlib.pyplot as plt
import numpy as np
import torch
from autoencoder import Autoencoder
from scripts.utils import load_data

# Hyperparameters (should match those used during training)
batch_size = 256
hidden_dim, hidden_layers = 150, 10
use_fraction = False
use_filtered = True
snp_fraction = 0.1  # which fraction of snps to randomly subsample

# Device configuration
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device_name}")
device = torch.device(device_name)
torch.set_default_device(device)
generator = torch.Generator(device=device)

# Load your data
_, train_dataloader, test_dataloader = load_data(batch_size, generator, use_filtered, use_fraction, snp_fraction)

# Load the saved model
input_dim = train_dataloader.dataset[0][0].shape[0]  # Assuming the dataset is a list of tuples (data, label)
model = Autoencoder(input_dim, hidden_dim, hidden_layers, 3).to(device)
model.load_state_dict(torch.load('../models/autoencoder_v2.pth'))
model.eval()

# Function to extract latent representations
def extract_latent_representations(dataloader, model):
    model.eval()
    latent_space = []
    labels = []
    with torch.no_grad():
        for features, label in dataloader:
            features = features.to(device)
            latent = model.encode(features)
            latent_space.append(latent.cpu().numpy())
            labels.append(label.cpu().numpy())
    latent_space = np.concatenate(latent_space, axis=0)
    labels = np.concatenate(labels, axis=0)
    return latent_space, labels

# Function to visualize latent representations
def visualize_latent_representations():
    # Extract latent representations for train and test datasets
    train_latent_space, train_labels = extract_latent_representations(train_dataloader, model)
    test_latent_space, test_labels = extract_latent_representations(test_dataloader, model)

    # Combine train and test data for visualization
    latent_space = np.concatenate((train_latent_space, test_latent_space), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)

    # 3D Scatter plot of the latent space
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(latent_space[:, 0], latent_space[:, 1], latent_space[:, 2], c=labels[:, 0], cmap='viridis', marker='o')
    plt.colorbar(sc, label='Label')
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_zlabel('Latent Dimension 3')
    plt.title('3D Scatter Plot of Latent Space')
    plt.show()

    maxs = latent_space.max(axis=0)
    mins = latent_space.min(axis=0)
    print(f"Max: {maxs}, Min: {mins}, Diffs: {maxs - mins}")

# Call the visualization function
visualize_latent_representations()
