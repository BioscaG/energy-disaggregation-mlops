from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from loguru import logger

from energy_dissagregation_mlops.model import Model
from energy_dissagregation_mlops.data import MyDataset


def train(
    preprocessed_folder: str = "data/processed",
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs: int = 3,
    num_workers: int = 2,
    device: str | None = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    preprocessed_folder = str(preprocessed_folder)

    logger.info(f"Starting training with device={device}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
    logger.debug(f"Using preprocessed folder: {preprocessed_folder}")

    # Dataset returns (x, y): x = mains [1, T], y = appliance [1, T]
    logger.info("Loading dataset...")
    dataset = MyDataset(
        data_path=Path("data/raw/ukdale.h5"),  # not used at runtime if preprocessed_folder is set
        preprocessed_folder=Path(preprocessed_folder),
        window_size=256,
        stride=256,
    )

    # Simple split (train/val) by index
    n = len(dataset)
    n_train = int(0.9 * n)
    n_val = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    logger.info(f"Dataset loaded: {n} total samples ({n_train} train, {n_val} val)")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    logger.info("Initializing model...")
    model = Model(window_size=1024).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    ckpt_path = Path("models")
    ckpt_path.mkdir(parents=True, exist_ok=True)
    best_file = ckpt_path / "best.pt"

    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Checkpoint will be saved to: {best_file}")

    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs} starting...")
        # ---- train ----
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            # x: [B, 1, T] (mains power)
            # y: [B, 1, T] (appliance power)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            y_hat = model(x)

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        logger.info(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_file,
            )
            logger.success(f"New best model saved to {best_file} (val_loss={best_val:.6f})")

    logger.success(f"Training complete! Best val_loss={best_val:.6f}")


if __name__ == "__main__":
    train()
