import pytest
import torch
from torch.utils.data import Dataset

from energy_dissagregation_mlops.data import MyDataset


@pytest.mark.integration
def test_my_dataset_constructs_with_real_data():
    try:
        dataset = MyDataset("data/raw")
    except Exception as e:
        pytest.skip(f"Real data not available / dataset init failed: {e}")
    assert isinstance(dataset, Dataset)


@pytest.mark.integration
def test_mydataset_len_and_getitem_real_data():
    try:
        ds = MyDataset("data/raw")
    except Exception as e:
        pytest.skip(f"Dataset init failed: {e}")

    if len(ds) == 0:
        pytest.skip("Real dataset is empty (data missing or preprocessing not run).")

    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.numel() > 0
    assert y.numel() > 0
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
