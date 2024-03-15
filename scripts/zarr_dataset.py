import numpy as np
import pandas as pd
import zarr
from torch.utils.data import Dataset, Subset


class ZarrDataset(Dataset):
    def __init__(self, zarr_file, zarr_path, label_file, label_cols, sample_transform=None, label_transform=None,
                 panda_kwargs=None):
        df = pd.read_csv(label_file, **(panda_kwargs or {}))
        self.labels = df[label_cols].values
        self.zarr = zarr.open(zarr_file, mode='r')[zarr_path]
        self.sample_transform = sample_transform
        self.label_transform = label_transform or (lambda row: row)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sample_transform(self.zarr[index]), self.label_transform(self.labels[index])

    def filter(self, label_criteria):
        mask = np.apply_along_axis(label_criteria, axis=1, arr=self.labels)
        indices = mask.nonzero()[0]
        return Subset(self, indices)
