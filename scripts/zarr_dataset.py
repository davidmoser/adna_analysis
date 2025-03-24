import numpy as np
import pandas as pd
import zarr
from torch.utils.data import Dataset, Subset


class ZarrDataset(Dataset):
    def __init__(self, zarr_file, zarr_path, label_file, label_cols, sample_transform=None, label_transform=None,
                 panda_kwargs=None):
        df = pd.read_csv(label_file, **(panda_kwargs or {}))
        self.labels = df[label_cols].values
        store = zarr.storage.ZipStore(zarr_file, mode="r")
        self.zarr = zarr.open_array(store, path=zarr_path, mode="r")
        if len(self.labels) != self.zarr.shape[0]:
            raise RuntimeError(f"Number of labels {len(self.labels)} does not match ZARR array length (axis 0) {self.zarr.shape[0]}")
        self.sample_transform = sample_transform or (lambda sample: sample)
        self.label_transform = label_transform or (lambda row: row)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sample_transform(self.zarr[index]), self.label_transform(self.labels[index])

    def filter(self, label_criteria):
        mask = np.apply_along_axis(label_criteria, axis=1, arr=self.labels)
        indices = mask.nonzero()[0]
        return Subset(self, indices)
