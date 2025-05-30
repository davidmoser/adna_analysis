# Script to train with autoencoder, down to a three dimensional space for color encoding
# We also load the labels for location and time to investigate the results

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from scripts.utils import log_system_usage
from scripts.utils import calculate_loss, load_data, plot_loss, snp_cross_entropy_loss, print_genotype_predictions
from autoencoder import Autoencoder

device_name = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device_name}")

device = torch.device(device_name)
torch.set_default_device(device)
generator = torch.Generator(device=device)

# Hyperparameters
batch_size = 256
learning_rate = 0.02
hidden_dim, hidden_layers = 150, 20
epochs = 100

# Load your data from a Zarr file
dataset, train_dataloader, test_dataloader = load_data(batch_size, generator, small=True, in_memory=True)

# Initialize the model, loss function, and optimizer
sample, _ = next(iter(dataset))
input_dim = 4 * len(sample)
print(f"Creating model, Input dimension: {input_dim}")
model = Autoencoder(input_dim, hidden_dim, hidden_layers, 3)
loss_function = snp_cross_entropy_loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
print("finished")

# Visualize


# Function to extract latent representations
def extract_latent_representations(dataloader, model):
    model.eval()
    latent_space = []
    labels = []
    with torch.no_grad():
        for features, label in dataloader:
            latent = model.encode(features)
            latent_space.append(latent.cpu().numpy())
            labels.append(label.cpu().numpy())
    latent_space = np.concatenate(latent_space, axis=0)
    labels = np.concatenate(labels, axis=0)
    return latent_space, labels


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
    sc = ax.scatter(latent_space[:, 0], latent_space[:, 1], latent_space[:, 2], c=labels[:, 0], cmap='viridis',
                    marker='o')
    plt.colorbar(sc, label='Label')
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_zlabel('Latent Dimension 3')
    plt.title('3D Scatter Plot of Latent Space')
    plt.show()

    maxs = latent_space.max(axis=0)
    mins = latent_space.min(axis=0)
    print(f"Max: {maxs}, Min: {mins}, Diffs: {maxs - mins}")

# Training loop
train_losses, test_losses = [], []

train_loss_previous = 0
train_loss_diff = 0
train_loss_variance = 0
for epoch in range(epochs):
    model.train()
    train_loss = 0
    number_batches = 0
    for feature_batch, _ in train_dataloader:
        print(".", end="")
        prediction = model(feature_batch)
        train_loss_obj = loss_function(prediction, feature_batch)
        train_loss += train_loss_obj.item()
        number_batches += 1

        optimizer.zero_grad()
        train_loss_obj.backward()
        optimizer.step()

    print("")
    scheduler.step()
    # Loss and change in loss
    train_loss /= number_batches
    train_losses.append(train_loss)
    train_loss_current_diff = train_loss_previous - train_loss
    train_loss_diff = (train_loss_diff * 9 + train_loss_current_diff) / 10
    # Validation loss
    test_loss = calculate_loss(model, test_dataloader, loss_function, invert_input=False, invert_output=True)
    test_losses.append(test_loss)
    # Print it out
    loss_scale = 1e4
    print(f'Epoch {epoch + 1}, T-Loss: {round(loss_scale * train_loss)}, V-Loss: {round(loss_scale * test_loss)}')
    log_system_usage()
    print_genotype_predictions(model, test_dataloader)
    train_loss_previous = train_loss

    visualize_latent_representations()

plot_loss(train_losses, test_losses, f'Color Autoencoder: Dimension: {hidden_dim}, Layers: {hidden_layers}, '
                                     f'Learning rate: {learning_rate}, Batch size: {batch_size}')

# Save the final model
#torch.save(model.state_dict(), '../models/autoencoder.pth')
