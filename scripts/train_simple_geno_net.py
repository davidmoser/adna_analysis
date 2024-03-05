import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import zarr
from torch.utils.data import TensorDataset, DataLoader

import anno
from simple_geno_net import SimpleGenoNet

# Hyperparameters
batch_size = 256
learning_rate = 0.0001
hidden_dim, hidden_layers = 50, 20
epochs = 100

# Load your data from a Zarr file
print("Preparing data")
locations = np.array(list(anno.get_locations().values()))  # s x 2
is_not_nan = ~np.isnan(locations).any(axis=1)
locations = locations[is_not_nan]

ages = np.array(list(anno.get_ages().values()))  # s
ages = ages[is_not_nan]

zarr_file = '../data/aadr_v54.1.p1_1240K_public_filtered.zarr'
z = zarr.open(zarr_file, mode='r')
genotypes = np.array(z['calldata/GT']).T
genotypes = genotypes[is_not_nan]

features = torch.Tensor(genotypes)  # s (samples) x n (SNPs)
labels = SimpleGenoNet.real_to_train(torch.Tensor(np.column_stack((ages, locations))))

# Split data into training and test sets
# features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
#                                                                             random_state=42)

# Create DataLoader instances for training and testing
dataset = TensorDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# train_dataset = TensorDataset(features_train, labels_train)
# test_dataset = TensorDataset(features_test, labels_test)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("finished")

# Initialize the model, loss function, and optimizer
print(f"Creating model, Input dimension: {features.shape[1]}")
input_dim = features.shape[1]
model = SimpleGenoNet(input_dim, hidden_dim, hidden_layers)
loss_function = nn.MSELoss()  # Using Mean Squared Error Loss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print("finished")


def print_sample(index):
    label = labels[[index]]
    prediction = model(features[[index]])
    loss = loss_function(label, prediction).item()
    label = SimpleGenoNet.train_to_real(label)
    prediction = SimpleGenoNet.train_to_real(prediction)
    print(f"Label: {label}, Prediction: {prediction}, Loss: {loss}")


# Training loop
train_losses, test_losses = [], []

train_loss_previous = 0
train_loss_average = 0
train_loss_variance = 0
for epoch in range(epochs):

    train_loss_scalar = 0
    for id_batch, (feature_batch, label_batch) in enumerate(dataloader):
        output = model(feature_batch)
        train_loss = loss_function(output, label_batch)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_loss_scalar = train_loss.item()

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
    train_losses.append(train_loss_scalar)
    train_loss_current_diff = train_loss_previous - train_loss_scalar
    train_loss_average = (train_loss_average * 9 + train_loss_current_diff) / 10
    train_loss_variance = (train_loss_variance * 9 + abs(train_loss_current_diff)) / 10
    loss_scale = 10e6
    print(f'Epoch {epoch + 1}, T-Loss: {round(loss_scale * train_loss_scalar)}, '
          f'T-Difference: {round(loss_scale * train_loss_average)}, T-Variance: {round(loss_scale * train_loss_variance)}')
    # print_sample(100)
    # print_sample(200)
    # print_sample(300)
    train_loss_previous = train_loss_scalar

plt.figure(figsize=(10, 5))
plt.plot(np.log10(train_losses), label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.title(f'Simple-Geno-Net: Dimension: {hidden_dim}, Layers: {hidden_layers}, '
          f'Learning rate: {learning_rate}, Batch size: {batch_size}')
plt.legend()
plt.show()

# Save the final model
torch.save(model.state_dict(), 'model_final.pth')
