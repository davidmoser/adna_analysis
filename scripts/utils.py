import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import anno
from simple_geno_net import SimpleGenoNet
from zarr_dataset import ZarrDataset


def use_device(device_name):
    print(f"Using device: {device_name}")
    device = torch.device(device_name)
    torch.set_default_device(device)
    generator = torch.Generator(device=device)
    return generator


def calculate_loss(model, dataloader, loss_function, invert=False):
    model.eval()
    loss = 0
    weight = 0
    for features, labels in dataloader:
        output = model(labels if invert else features)
        loss += loss_function(output, features if invert else labels).item() * len(features)
        weight += len(features)
    return loss / weight


def load_data(batch_size, generator, use_filtered, use_fraction, snp_fraction):
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
    return dataset, train_dataloader, test_dataloader


def plot_loss(train_losses, test_losses, title):
    plt.figure(figsize=(10, 5))
    plt.plot(np.log10(train_losses), label='Training Loss')
    plt.plot(np.log10(test_losses), label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title(title)
    plt.legend()
    plt.show()
