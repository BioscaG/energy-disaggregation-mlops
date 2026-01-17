import pytest
import torch
from torch.utils.data import Dataset


class SyntheticNILMDataset(Dataset):
    """
    Simple deterministic synthetic dataset for unit tests.
    Produces (x, y) pairs compatible with your Model forward signature.
    """

    def __init__(self, n=64, window=1024, seed=42):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, 1, window, generator=g)
        y = 0.5 * x + 0.05 * torch.randn(n, 1, window, generator=g)
        self.x = x.float()
        self.y = y.float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


@pytest.fixture(scope="session")
def device():
    """
    CPU by default; uses CUDA if available.
    Kept as a fixture so tests don't hardcode device logic.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture()
def synthetic_ds():
    """
    Synthetic dataset fixture for fast, CI-safe unit tests.
    """
    return SyntheticNILMDataset()
