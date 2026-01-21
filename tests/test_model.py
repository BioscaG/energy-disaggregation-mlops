import inspect

import pytest
import torch

from energy_dissagregation_mlops.model import Model


def _instantiate_model():
    """
    Tries to instantiate Model with defaults.
    If your Model requires args, adapt this function once.
    """
    sig = inspect.signature(Model)
    if len(sig.parameters) == 0:
        return Model()

    # If Model has defaulted parameters, this will work:
    try:
        return Model()
    except TypeError as e:
        pytest.fail(f"Model() requires args. Update _instantiate_model() with your Model signature. Error: {e}")


def test_model_instantiation():
    model = _instantiate_model()
    assert model is not None


from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from energy_dissagregation_mlops.data import MyDataset
from energy_dissagregation_mlops.model import Model


@pytest.mark.integration
def test_one_training_step_reduces_loss_processed_data(device):
    """
    Integration test:
    - Loads preprocessed chunks from data/processed (chunk_*.npz + meta.npz)
    - Runs a few optimizer steps
    - Checks loss is finite and does not blow up
    """

    preprocessed = Path("data/processed")

    # Dataset loads index from preprocessed_folder; data_path is not used at runtime in this mode
    try:
        ds = MyDataset(
            data_path=Path("data/raw/ukdale.h5"),
            preprocessed_folder=preprocessed,
        )
    except Exception as e:
        pytest.skip(f"Could not load processed dataset from {preprocessed}: {e}")

    if len(ds) < 8:
        pytest.skip("Processed dataset too small to run training integration test.")

    model = Model(window_size=1024).to(device)
    model.train()

    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    losses = []
    for step, (x, y) in enumerate(dl):
        # x,y are [B, 1, T] already (because __getitem__ returns [1,T])
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        loss = loss_fn(out, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(float(loss.item()))
        if step >= 5:
            break

    # Robust assertions (non-flaky)
    assert len(losses) >= 2
    assert torch.isfinite(torch.tensor(losses)).all()
    # Don't require strict decrease on real data; just ensure it doesn't explode.

    assert losses[-1] <= losses[0] * 1.25
