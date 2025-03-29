import torch
from torch.utils.data import Dataset, DataLoader


class TestDataset(Dataset):
    def __init__(self, n, size):
        self.n = n          # Length of input vectors
        self.size = size    # Number of samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.zeros(self.n)
        y = torch.rand(3)  # 3 random numbers ∈ [0, 1)
        return x, y