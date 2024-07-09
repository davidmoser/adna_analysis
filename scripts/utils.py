import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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


def calculate_loss(model, dataloader, loss_function, invert_input=False, invert_output=False):
    model.eval()
    loss = 0
    number_batches = 0
    for features, labels in dataloader:
        output = model(labels if invert_input else features)
        loss += loss_function(output, features if invert_output else labels).item()
        number_batches += 1
    return loss / number_batches


def load_data(batch_size, generator, use_filtered, use_fraction, snp_fraction, label_filter=None, flattened=True):
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

    def to_one_hot(genotypes, num_classes=4):
        # Convert the genotypes tensor to a new tensor with mapped values
        genotypes_mapped = torch.full_like(genotypes, -1, dtype=torch.long)

        # Map the values using vectorized operations
        genotypes_mapped = torch.where(genotypes == -2, 0, genotypes_mapped)
        genotypes_mapped = torch.where(genotypes == 0, 1, genotypes_mapped)
        genotypes_mapped = torch.where(genotypes == 1, 2, genotypes_mapped)
        genotypes_mapped = torch.where(genotypes == 2, 3, genotypes_mapped)

        # Use torch.nn.functional.one_hot to create the one-hot matrix
        one_hot_matrix = F.one_hot(genotypes_mapped, num_classes=num_classes)

        return one_hot_matrix

    def transform_sample(zarr_genotype):
        genotype = torch.tensor(zarr_genotype, dtype=torch.float32)
        filtered_genotype = genotype[snp_indices] if use_fraction else genotype
        one_hot_genotype = to_one_hot(filtered_genotype).to(dtype=torch.float32)
        return one_hot_genotype.view(-1) if flattened else one_hot_genotype

    dataset.sample_transform = transform_sample

    dataset = dataset.filter(label_filter or (lambda label: np.all(~np.isnan(label)) and 100 < label[0] <= 10000))

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


# Optimized custom cross-entropy loss for 4D one-hot vectors
def snp_cross_entropy_loss(output, target):
    # Reshape output and target tensors
    output = output.view(-1, 4)
    target = target.view(-1, 4)

    # Convert one-hot target to class indices
    target_indices = torch.argmax(target, dim=1)

    # Compute cross-entropy loss
    loss = torch.nn.CrossEntropyLoss(reduction='mean')(output, target_indices)

    return loss


def print_genotype_predictions(model, dataloader, invert=False):
    print_genotype_prediction(model, dataloader, 0, invert)
    print_genotype_prediction(model, dataloader, 1, invert)
    print_genotype_prediction(model, dataloader, 2, invert)


def print_genotype_prediction(model, dataloader, index, invert=False):
    model.eval()
    features, labels = next(iter(dataloader))
    model_input = labels if invert else features
    logits = model(model_input[[index]]).view(-1, 4)
    p = torch.nn.functional.softmax(logits, dim=1)

    def print_counts(genotypes):
        counts = genotypes.view(-1, 4).sum(dim=0)
        print(f"Undet: {counts[0]:.0f}, Homozyg.Ref.: {counts[1]:.0f}, "
              f"Heterozyg.: {counts[2]:.0f}, Homozyg.Alt.: {counts[3]:.0f}")

    print("Original ", end='')
    print_counts(features[[index]])
    print("Prediction ", end='')
    print_counts(p.detach())
