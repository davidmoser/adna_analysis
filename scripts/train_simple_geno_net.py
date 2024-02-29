import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import zarr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import anno
from simple_geno_net import SimpleGenoNet

# Load your data from a Zarr file
print("Loading data")
locations = np.array(list(anno.get_locations().values()))  # s x 2
is_not_nan = ~np.isnan(locations).any(axis=1)
locations = locations[is_not_nan]
locations = np.array([[-90, 0], [0, 0], [90, 0]])

ages = np.array(list(anno.get_ages().values()))  # s
ages = ages[is_not_nan]
ages = [100, 1000, 10000]

zarr_file = '../data/aadr_v54.1.p1_1240K_public_eigenstrat.zarr'
z = zarr.open(zarr_file, mode='r')
genotypes = np.array(z['calldata/GT']).T
genotypes = genotypes[is_not_nan]
genotypes = [[0, 0, 2], [0, 2, 0], [2, 0, 0]]

features = torch.Tensor(genotypes)  # s (samples) x n (SNPs)
labels = torch.Tensor(np.column_stack((ages, locations)))
print("finished")

# Split data into training and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,
                                                                            random_state=42)

# Create DataLoader instances for training and testing
train_dataset = TensorDataset(features_train, labels_train)
test_dataset = TensorDataset(features_test, labels_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
print("Creating model")
n, m, k = features.shape[1], 3, 0  # Assuming 'm' and 'k' as previously defined; 'n' is inferred from data
model = SimpleGenoNet(n, m, k)
criterion = nn.MSELoss()  # Using Mean Squared Error Loss for regression tasks
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
print("finished")


# Function to compute loss over a dataset
def compute_loss(data_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(data_loader)


# Training loop
epochs = 10
train_losses, test_losses = [], []

for epoch in range(epochs):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    train_loss = compute_loss(train_loader)
    test_loss = compute_loss(test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

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

    print(f'Epoch {epoch + 1}, Training Loss: {train_loss}, Test Loss: {test_loss}')

# Save the final model
torch.save(model.state_dict(), 'model_final.pth')
