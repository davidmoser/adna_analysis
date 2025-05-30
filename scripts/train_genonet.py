# Script to train with genonet, given SNPs predict age and location
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from scripts.genonet import Genonet
from scripts.utils import log_system_usage, to_one_hot
from scripts.utils import use_device, calculate_loss, plot_loss, load_data

device_name = "cuda" if torch.cuda.is_available() else "cpu"
generator = use_device(device_name)

def train_genonet(epochs, learning_rate, batch_size, hidden_dim, hidden_layers, small=True, schedule="cosine", verbose=False):
    # Load your data from a Zarr file
    dataset, train_dataloader, test_dataloader = load_data(batch_size, generator, in_memory=True, small=small)

    # Initialize the model, loss function, and optimizer
    sample, label = dataset[0]
    print(f"Creating model, Input dimension: {len(sample)}")
    input_dim = 4 * len(sample)
    model = Genonet(input_dim, 3, hidden_dim, hidden_layers, first_fun=to_one_hot, batch_norm=True)
    model = model.to(torch.float32)

    loss_function = nn.MSELoss()  # Using Mean Squared Error Loss for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif schedule == "constant":
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1)
    else:
        raise ValueError(f"Unknown scheduler {schedule}")
    print("finished")


    def print_sample(index, dataloader):
        model.eval()
        with torch.no_grad():
            test_features, test_labels = next(iter(dataloader))
            label = test_labels[[index]]
            prediction = model(test_features[[index]])
            loss = loss_function(label, prediction).item()
            label = Genonet.train_to_real(label)
            prediction = Genonet.train_to_real(prediction)
            print(f"Label: {label}, Prediction: {prediction}, Loss: {loss}")


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
            output = model(feature_batch)
            train_loss_obj = loss_function(output, label_batch)
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
        test_loss = calculate_loss(model, test_dataloader, loss_function)
        test_losses.append(test_loss)
        # Print it out
        loss_scale = 1e7
        print(f'Epoch {epoch + 1}, T-Loss: {round(loss_scale * train_loss)}, V-Loss: {round(loss_scale * test_loss)}')
        if verbose:
            log_system_usage()
            print_sample(0, test_dataloader)
            print_sample(1, test_dataloader)
            print_sample(0, train_dataloader)
            print_sample(1, train_dataloader)
        train_loss_previous = train_loss

    return model, train_losses, test_losses


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.002
    hidden_dim, hidden_layers = 150, 20
    epochs = 10

    model, train_losses, test_losses = train_genonet(epochs, learning_rate, batch_size, hidden_dim, hidden_layers, schedule="constant", small=True)

    plot_loss(train_losses, test_losses, f'Simple-Geno-Net: Dimension: {hidden_dim}, Layers: {hidden_layers}, '
                                         f'Learning rate: {learning_rate}, Batch size: {batch_size}')