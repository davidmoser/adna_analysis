# The traditional approach to scale down number of SNPs for further analysis:
# Amongst a random sample of SNPs it finds the unlinked SNPs
# This is the method used to create the 'filtered' dataset

import allel
import dask.array as da
import numpy as np
import zarr

# Path to your VCF file
genotypes_all_path = '../data/aadr_v54.1.p1_1240K_public_all.zarr'
calldata_path = 'calldata/GT'
samples_path = 'samples'
genotypes_filtered_path = '../data/aadr_v54.1.p1_1240K_public_filtered.zarr'


def print_count(gs):
    print(f"Genotype count: {gs.shape[0]}")


# Load genotype data from VCF
genotypes = da.from_zarr(genotypes_all_path + '/' + calldata_path)
samples = da.from_zarr(genotypes_all_path + '/' + samples_path)
print_count(genotypes)

# randomly choose some genotypes, locate_unlink is too slow otherwise
print("Choosing randomly")
n = 10000
vidx = np.random.choice(genotypes.shape[0], n, replace=False)
vidx.sort()
genotypes = genotypes[vidx].compute()
print_count(genotypes)

# find unlinked genotypes
print("Finding unlinked")
matches = allel.locate_unlinked(genotypes, size=500, step=200, threshold=.1, blen=1000)
genotypes = genotypes[matches]
print_count(genotypes)

# remove genotypes where all calls are the same
print("Removing uniform")
keep_genotypes = []
for index, genotype in enumerate(genotypes):
    if np.any(genotype != genotype[0]):
        keep_genotypes.append(index)

genotypes = genotypes[keep_genotypes]
print_count(genotypes)

# Save in new zarr
print(f"Saving to {genotypes_filtered_path}")
zarr_store = zarr.DirectoryStore(genotypes_filtered_path)
zarr.save_array(zarr_store, genotypes, path=calldata_path)
