import numpy as np
import pandas as pd
import zarr
from torch.utils.data import Dataset, Subset


class ZarrDataset(Dataset):
    def __init__(self, anno_file, anno_cols, zarr_file, zarr_path):
        df = pd.read_csv(anno_file, sep='\t', quotechar='$', low_memory=False, on_bad_lines='warn', na_values='..')
        self.labels = df[anno_cols].values
        self.zarr = zarr.open(zarr_file, mode='r')[zarr_path]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.zarr[index], self.labels[index]

    def filter_nan(self):
        mask = np.all(~np.isnan(self.labels), axis=1)
        indices = mask.nonzero()[0]
        return Subset(self, indices)
