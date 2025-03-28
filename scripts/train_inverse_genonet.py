# Script to train inverse genonet, given age and location predict some "expected" SNPs
import torch
import torch.optim as optim

from scripts.utils import log_system_usage
from scripts.utils import calculate_loss, load_data, use_device, plot_loss, snp_cross_entropy_loss, \
    print_genotype_predictions
from genonet import Genonet

device_name = "cuda" if torch.cuda.is_available() else "cpu"
generator = use_device(device_name)

# Hyperparameters
batch_size = 256
learning_rate = 0.002
hidden_dim, hidden_layers = 150, 10
epochs = 200

# Load your data from a Zarr file
dataset, train_dataloader, test_dataloader = load_data(batch_size, generator, in_memory=True, small=True)

# Initialize the model, loss function, and optimizer
genotypes, label = dataset[0]
print(f"Creating model, Input dimension: 3, Output dimension: {len(genotypes)}")
nb_snps = len(genotypes)
output_dim = 4 * nb_snps
model = Genonet(3, output_dim, hidden_dim, hidden_layers, final_fun=lambda x: x)
loss_function = snp_cross_entropy_loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ConstantLR(optimizer)
print("finished")


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
    test_loss = calculate_loss(model, test_dataloader, loss_function, invert_input=True, invert_output=True)
    test_losses.append(test_loss)
    # Print it out
    loss_scale = 1e4
    print(f'Epoch {epoch + 1}, T-Loss: {round(loss_scale * train_loss)}, V-Loss: {round(loss_scale * test_loss)}')
    log_system_usage()
    print_genotype_predictions(model, test_dataloader, invert=True)
    train_loss_previous = train_loss

plot_loss(train_losses, test_losses, f'Inverse-Genonet: Dimension: {hidden_dim}, Layers: {hidden_layers}, '
                                     f'Learning rate: {learning_rate}, Batch size: {batch_size}')

# Save the final model
torch.save(model.state_dict(), '../models/inverse_genonet.pth')
