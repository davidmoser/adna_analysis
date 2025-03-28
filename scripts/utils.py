import subprocess

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import anno
from genonet import Genonet
from zarr_dataset import ZarrDataset


def use_device(device_name):
    print(f"Using device: {device_name}")
    device = torch.device(device_name)
    torch.set_default_device(device)
    generator = torch.Generator(device=device)
    return generator


def calculate_loss(model, dataloader, loss_function, invert_input=False, invert_output=False):
    model.eval()
    with torch.no_grad():
        loss = 0
        number_batches = 0
        for features, labels in dataloader:
            output = model(labels if invert_input else features)
            loss += loss_function(output, features if invert_output else labels).item()
            number_batches += 1
        return loss / number_batches


def load_data(batch_size, generator, small=False, label_filter=None, in_memory=False):
    print("Preparing data")
    dataset = ZarrDataset(
        zarr_file='../../adna_retrieval_conversion/zarr/v62.0_1240k_public_small_ind_chunked.zarr.zip' if small
        else '../../adna_retrieval_conversion/zarr/v62.0_1240k_public_complete_ind_chunked.zarr.zip',
        zarr_path='calldata/GT',
        sample_transform=None,
        label_file='../../adna_retrieval_conversion/vcf/v62.0_1240k_public_small.anno' if small
        else '../../adna_retrieval_conversion/ancestrymap/v62.0_1240k_public.anno',
        label_cols=[anno.age_colname, anno.long_col, anno.lat_col],
        label_transform=lambda lbl: torch.tensor(Genonet.real_to_train_single(lbl), dtype=torch.float32),
        panda_kwargs={'sep': '\t', 'quotechar': '$', 'low_memory': False, 'on_bad_lines': 'warn', 'na_values': '..'},
        in_memory=in_memory,
    )
    print(f"Dataset length {len(dataset)}")

    dataset = dataset.filter(label_filter or (lambda label: np.all(~np.isnan(label)) and 100 < label[0] <= 10000))
    print(f"Filtered dataset {len(dataset)}")

    # Create DataLoader instances for training and testing
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1], generator=generator)
    print(f"Train dataset {len(train_dataset)}")
    print(f"Test dataset {len(test_dataset)}")
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


def snp_cross_entropy_loss(output, target_indices):
    output = output.view(output.shape[0], 4, -1)
    return torch.nn.CrossEntropyLoss(reduction='mean')(output, target_indices.long())


def print_genotype_predictions(model, dataloader, invert=False):
    model.eval()
    with torch.no_grad():
        genotypes, labels = next(iter(dataloader))
        genotypes_oh = to_one_hot(genotypes[:3])
        print_genotype_prediction(model, genotypes_oh, labels, 0, invert)
        print_genotype_prediction(model, genotypes_oh, labels, 1, invert)
        print_genotype_prediction(model, genotypes_oh, labels, 2, invert)


def print_genotype_prediction(model, features, labels, index, invert=False):
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


def log_system_usage():
    # RAM Usage
    ram_usage = psutil.virtual_memory().percent  # Get the system RAM usage percentage
    log = f"System RAM usage: {ram_usage}%"

    if torch.cuda.is_available():
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader']
        )
        log += f", GPU usage: {output.decode().strip()}%"

        # GPU RAM Usage
        gpu_ram_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Convert bytes to GB
        gpu_ram_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)  # Convert bytes to GB
        log += f", GPU RAM allocated: {gpu_ram_allocated:.2f} GB"
        log += f", GPU RAM reserved: {gpu_ram_reserved:.2f} GB"

    print(log)


# converts a batch of index genotypes (0: both reference, 1: one alternate, 2: both alternate, 3: no call)
# to one hot genotypes ([1 0 0 0]: both reference, [0 1 0 0]: one alternate, ...)
def to_one_hot(index_genotypes):
    one_hot_genotypes = F.one_hot(index_genotypes.long(), num_classes=4).to(dtype=torch.float32)
    return one_hot_genotypes.view(index_genotypes.shape[0], -1)