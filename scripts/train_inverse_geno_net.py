# Script to train inverse genonet, given age and location predict some "expected" SNPs

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader

import anno
from scripts.log_memory import log_memory_usage
from simple_geno_net import SimpleGenoNet
from zarr_dataset import ZarrDataset

# device_name = "cuda" if torch.cuda.is_available() else "cpu"
device_name = 'cuda'
print(f"Using device: {device_name}")

device = torch.device(device_name)
torch.set_default_device(device)
generator = torch.Generator(device=device)

# Hyperparameters
batch_size = 256
learning_rate = 0.001
hidden_dim, hidden_layers = 100, 20
epochs = 30
use_fraction = False
use_filtered = True
snp_fraction = 0.1  # which fraction of snps to randomly subsample

# Load your data from a Zarr file
print("Preparing data")
dataset = ZarrDataset(
    zarr_file='../data/aadr_v54.1.p1_1240K_public_filtered_arranged.zarr' if use_filtered else '../data/aadr_v54.1.p1_1240K_public_all_arranged.zarr',
    zarr_path='calldata/GT',
    sample_transform=None,
    label_file='../data/aadr_v54.1.p1_1240K_public.anno',
    label_cols=[anno.age_col, anno.long_col, anno.lat_col],
    label_transform=lambda lbl: torch.tensor(SimpleGenoNet.real_to_train_single(lbl), dtype=torch.float32),
    panda_kwargs={'sep': '\t', 'quotechar': '$', 'low_memory': False, 'on_bad_lines': 'warn', 'na_values': '..'}
)

total_snps = dataset.zarr.shape[1]
snp_indices = np.random.choice(total_snps, size=int(total_snps * snp_fraction), replace=False)


def transform_sample(zarr_genotype):
    genotype = torch.tensor(zarr_genotype, dtype=torch.float32)
    return genotype[snp_indices] if use_fraction else genotype


dataset.sample_transform = transform_sample

dataset = dataset.filter(lambda label: np.all(~np.isnan(label)) and 100 < label[0] <= 10000)

# Create DataLoader instances for training and testing
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1], generator=generator)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator=generator)
print("finished")

# Initialize the model, loss function, and optimizer
sample, label = next(iter(dataset))
print(f"Creating model, Input dimension: 3, Output dimension: {len(sample)}")
output_dim = len(sample)
model = SimpleGenoNet(3, output_dim, hidden_dim, hidden_layers)
loss_function = nn.MSELoss()  # Using Mean Squared Error Loss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ExponentialLR(optimizer, gamma=0.95)
print("finished")


def calculate_loss(dataloader):
    model.eval()
    loss = 0
    weight = 0
    for features, labels in dataloader:
        output = model(labels)
        loss += loss_function(output, features).item() * len(features)
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
    number_batches = 0
    for feature_batch, label_batch in train_dataloader:
        print(".", end="")
        output = model(label_batch)
        train_loss_obj = loss_function(output, feature_batch)
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
    # Validation loss
    test_loss = calculate_loss(test_dataloader)
    test_losses.append(test_loss)
    # Print it out
    loss_scale = 1e4
    print(f'Epoch {epoch + 1}, T-Loss: {round(loss_scale * train_loss)}, V-Loss: {round(loss_scale * test_loss)}')
    log_memory_usage()
    train_loss_previous = train_loss

plt.figure(figsize=(10, 5))
plt.plot(np.log10(train_losses), label='Training Loss')
plt.plot(np.log10(test_losses), label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.title(f'Inverse-Geno-Net: Dimension: {hidden_dim}, Layers: {hidden_layers}, '
          f'Learning rate: {learning_rate}, Batch size: {batch_size}')
plt.legend()
plt.show()

# Save the final model
torch.save(model.state_dict(), '../models/inverse_geno_net.pth')