import allel
import dask.array as da
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import anno

# Path to your VCF file
genotypes_path = '../data/aadr_v54.1.p1_1240K_public_eigenstrat_all.zarr/calldata/GT'
samples_path = '../data/aadr_v54.1.p1_1240K_public_eigenstrat_all.zarr/samples'

# Load genotype data from VCF
genotypes = da.from_zarr(genotypes_path)
samples = da.from_zarr(samples_path)
print(f"Genotype count: {genotypes.shape[0]}")

# randomly choose some genotypes, locate_unlink is too slow otherwise
print("Choosing randomly")
n = 10000
vidx = np.random.choice(genotypes.shape[0], n, replace=False)
vidx.sort()
genotypes = genotypes[vidx].compute()
print(f"Genotype count: {genotypes.shape[0]}")

# find unlinked genotypes
print("Finding unlinked")
matches = allel.locate_unlinked(genotypes, size=500, step=200, threshold=.1, blen=1000)
genotypes = genotypes[matches]
print(f"Genotype count: {genotypes.shape[0]}")

# remove genotypes where all calls are the same
print("Removing uniform")
keep_genotypes = []
for index, genotype in enumerate(genotypes):
    if np.any(genotype != genotype[0]):
        keep_genotypes.append(index)

genotypes = genotypes[keep_genotypes]
print(f"Genotype count: {genotypes.shape[0]}")

# Perform PCA
print("Performing PCA")
coords, model = allel.pca(genotypes, n_components=4, copy=False)

# Plot PC1 vs. PC2
print("Plotting")

# Normalize the age data to range between 0 and 1
age_map = anno.get_ages()
ages = [min(age_map[sample], 10000) for sample in samples.compute()]
norm = mcolors.Normalize(vmin=np.min(ages), vmax=np.max(ages))

# Choose a colormap
colormap = plt.cm.viridis

# Map the normalized age values to colors
colors = colormap(norm(ages))

# Create a scatter plot with the mapped colors
fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# PC1 vs PC2
sc1 = axs[0].scatter(coords[:, 0], coords[:, 1], c=ages, cmap=colormap, norm=norm, s=1, linewidth=0)
axs[0].set_title('PC1 vs PC2')
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')

# PC1 vs PC2
sc2 = axs[1].scatter(coords[:, 2], coords[:, 3], c=ages, cmap=colormap, norm=norm, s=1, linewidth=0)
axs[1].set_title('PC3 vs PC4')
axs[1].set_xlabel('PC3')
axs[1].set_ylabel('PC4')

# Adjust layout before adding colorbar
fig.subplots_adjust(right=0.8)  # Adjust this value as needed to create space for the colorbar

# Create a single colorbar for the figure
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colormap), cax=cbar_ax, label='Age')

#plt.show()
plt.savefig('../results/principal_components_large.png')
plt.clf()
