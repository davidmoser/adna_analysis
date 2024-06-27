import matplotlib.pyplot as plt
import numpy as np

from scripts.utils import use_device, load_data

use_filtered = True
min_age = 100
max_age = 10000

# Load the Zarr array
generator = use_device("cpu")
zarr_data, _, _ = load_data(1000, generator, use_filtered, False, 0,
                            label_filter=lambda label: np.all(~np.isnan(label)) and min_age < label[0] <= max_age)

# Get the shape of the array
num_samples = zarr_data.__len__()
print(f"Number of samples {num_samples}")

# Initialize an array to store the counts
counts = np.zeros((num_samples, 4), dtype=int)

# Iterate through the samples
i = 0
for snps, labels in zarr_data:
    snps = snps.numpy()
    counts[i, 0] = np.sum(snps == -2)  # Undetermined (-2)
    counts[i, 1] = np.sum(snps == 0)  # Homozygous Reference (0)
    counts[i, 2] = np.sum(snps == 1)  # Heterozygous (1)
    counts[i, 3] = np.sum(snps == 2)  # Homozygous Alternative (2)
    i += 1

# Plot histograms for each possibility
labels = ['Undetermined (-2)', 'Homozygous Ref (0)', 'Heterozygous (1)', 'Homozygous Alt (2)']
for i in range(4):
    plt.figure()
    plt.hist(counts[:, i], bins=30, alpha=0.75)
    plt.title(f'Samples between {min_age} years and {max_age} years ago. {labels[i]} Counts')
    plt.xlabel('Count')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
