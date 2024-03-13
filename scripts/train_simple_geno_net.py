import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import zarr
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader

import anno
from simple_geno_net import SimpleGenoNet

# Hyperparameters
batch_size = 256
learning_rate = 0.0002
hidden_dim, hidden_layers = 50, 20
epochs = 100

# Load your data from a Zarr file
print("Preparing data")
locations = np.array(list(anno.get_locations().values()))  # s x 2
is_not_nan = ~np.isnan(locations).any(axis=1)
ages = np.array(list(anno.get_ages().values()))  # s
is_old = ages > 100
filter = is_not_nan & is_old

locations = locations[filter]
ages = ages[filter]

zarr_file = '../data/aadr_v54.1.p1_1240K_public_filtered.zarr'
z = zarr.open(zarr_file, mode='r')
genotypes = np.array(z['calldata/GT']).T
genotypes = genotypes[filter]

all_features = torch.Tensor(genotypes)  # s (samples) x n (SNPs)
all_labels = SimpleGenoNet.real_to_train(torch.Tensor(np.column_stack((ages, locations))))

# Create DataLoader instances for training and testing
train_dataset = TensorDataset(all_features, all_labels)
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
print("finished")

# Initialize the model, loss function, and optimizer
print(f"Creating model, Input dimension: {all_features.shape[1]}")
input_dim = all_features.shape[1]
model = SimpleGenoNet(input_dim, hidden_dim, hidden_layers)
loss_function = nn.MSELoss()  # Using Mean Squared Error Loss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ExponentialLR(optimizer, gamma=0.9)
print("finished")


def print_sample(index):
    model.eval()
    test_features, test_labels = next(iter(test_dataloader))
    label = test_labels[[index]]
    prediction = model(test_features[[index]])
    loss = loss_function(label, prediction).item()
    label = SimpleGenoNet.train_to_real(label)
    prediction = SimpleGenoNet.train_to_real(prediction)
    print(f"Label: {label}, Prediction: {prediction}, Loss: {loss}")


def calculate_loss(dataloader):
    model.eval()
    loss = 0
    weight = 0
    for features, labels in dataloader:
        output = model(features)
        loss += loss_function(output, labels).item() * len(features)
        weight += len(features)
    return loss / weight


# Training loop
train_losses, test_losses = [], []

train_loss_previous = 0
train_loss_diff = 0
train_loss_variance = 0
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for feature_batch, label_batch in train_dataloader:
        output = model(feature_batch)
        train_loss_obj = loss_function(output, label_batch)

        optimizer.zero_grad()
        train_loss_obj.backward()
        optimizer.step()

        # Update plot
        # if epoch % 1 == 0:
        #     plt.figure(figsize=(10, 5))
        #     plt.plot(train_losses, label='Training Loss')
        #     plt.plot(test_losses, label='Test Loss')
        #     plt.xlabel('Epoch')
        #     plt.ylabel('Loss')
        #     plt.title('Training and Test Loss')
        #     plt.legend()
        #     plt.show()
    scheduler.step()
    # Loss and change in loss
    train_loss = calculate_loss(train_dataloader)
    train_losses.append(train_loss)
    train_loss_current_diff = train_loss_previous - train_loss
    train_loss_diff = (train_loss_diff * 9 + train_loss_current_diff) / 10
    # Validation loss
    test_loss = calculate_loss(test_dataloader)
    test_losses.append(test_loss)
    # Print it out
    loss_scale = 10e6
    print(f'Epoch {epoch + 1}, T-Loss: {round(loss_scale * train_loss)}, '
          f'T-Difference: {round(loss_scale * train_loss_diff)}, V-Loss: {round(loss_scale * test_loss)}')
    print_sample(0)
    print_sample(1)
    print_sample(2)
    train_loss_previous = train_loss

plt.figure(figsize=(10, 5))
plt.plot(np.log10(train_losses), label='Training Loss')
plt.plot(np.log10(test_losses), label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.title(f'Simple-Geno-Net: Dimension: {hidden_dim}, Layers: {hidden_layers}, '
          f'Learning rate: {learning_rate}, Batch size: {batch_size}')
plt.legend()
plt.show()

# Save the final model
torch.save(model.state_dict(), 'model_final.pth')
