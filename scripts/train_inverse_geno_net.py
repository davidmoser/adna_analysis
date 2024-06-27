# Script to train inverse genonet, given age and location predict some "expected" SNPs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from scripts.log_memory import log_memory_usage
from scripts.utils import calculate_loss, load_data, use_device, plot_loss
from simple_geno_net import SimpleGenoNet

# device_name = "cuda" if torch.cuda.is_available() else "cpu"
generator = use_device("cuda")

# Hyperparameters
batch_size = 256
learning_rate = 0.001
hidden_dim, hidden_layers = 100, 20
epochs = 30
use_fraction = False
use_filtered = True
snp_fraction = 0.1  # which fraction of snps to randomly subsample

# Load your data from a Zarr file
dataset, train_dataloader, test_dataloader = load_data(batch_size, generator, use_filtered, use_fraction, snp_fraction)

# Initialize the model, loss function, and optimizer
sample, label = next(iter(dataset))
print(f"Creating model, Input dimension: 3, Output dimension: {len(sample)}")
output_dim = len(sample)
model = SimpleGenoNet(3, output_dim, hidden_dim, hidden_layers, final_fun=lambda x: 2 * torch.tanh(x))
loss_function = nn.MSELoss()  # Using Mean Squared Error Loss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ExponentialLR(optimizer, gamma=0.95)
print("finished")


def print_sample(index):
    model.eval()
    test_features, test_labels = next(iter(test_dataloader))
    # label = test_labels[[index]]
    # label = SimpleGenoNet.train_to_real(label)
    prediction = model(test_labels[[index]])

    def count(genotypes, value):
        return np.sum((value - 0.5 < genotypes) & (genotypes <= value + 0.5))

    def print_counts(genotypes):
        undet_count = count(genotypes, -2)
        homozygous_ref = count(genotypes, 0)
        heterozygous = count(genotypes, 1)
        homozygous_alt = count(genotypes, 2)
        out_count = genotypes.shape[1] - undet_count - homozygous_ref - heterozygous - homozygous_alt
        print(f"Undet: {undet_count}, Homozyg.Ref.: {homozygous_ref}, "
              f"Heterozyg.: {heterozygous}, Homozyg.Alt.: {homozygous_alt}, Outside: {out_count}")

    print("Original ", end='')
    print_counts(test_features[[index]].cpu().numpy())
    print("Prediction ", end='')
    print_counts(prediction.detach().cpu().numpy())


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
    test_loss = calculate_loss(model, test_dataloader, loss_function, invert=True)
    test_losses.append(test_loss)
    # Print it out
    loss_scale = 1e4
    print(f'Epoch {epoch + 1}, T-Loss: {round(loss_scale * train_loss)}, V-Loss: {round(loss_scale * test_loss)}')
    log_memory_usage()
    print_sample(0)
    print_sample(1)
    print_sample(2)
    train_loss_previous = train_loss

plot_loss(train_losses, test_losses, f'Inverse-Geno-Net: Dimension: {hidden_dim}, Layers: {hidden_layers}, '
                                     f'Learning rate: {learning_rate}, Batch size: {batch_size}')

# Save the final model
torch.save(model.state_dict(), '../models/inverse_geno_net.pth')
