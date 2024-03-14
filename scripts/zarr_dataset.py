import pandas
import zarr
from torch.utils.data import Dataset


class ZarrDataset(Dataset):
    def __init__(self, anno_file, zarr_file, zarr_path):
        self.annotations = pandas.read_csv(anno_file)
        self.zarr = zarr.open(zarr_file, mode='r')[zarr_path]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        return self.zarr[index], self.annotations.iloc[index]
