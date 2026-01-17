
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from energy_dissagregation_mlops.model import Model
from energy_dissagregation_mlops.data import MyDataset

def evaluate(
    preprocessed_folder: str = "data/processed",
    checkpoint_path: str = "models/best.pt",
    batch_size: int = 32,
    device: str | None = None,
    plot_results: bool = True
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting evaluation on {device}...")

    logger.info(f"Loading test dataset from: {preprocessed_folder}")
    dataset = MyDataset(
        data_path=Path("data/raw/ukdale.h5"),
        preprocessed_folder=Path(preprocessed_folder),
    )

    n = len(dataset)
    n_train = int(0.9 * n)
    _, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n - n_train],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Test set size: {len(test_ds)} samples")

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logger.info(f"Loading model checkpoint: {checkpoint_path}")
    model = Model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    logger.debug(f"Checkpoint info: epoch={checkpoint.get('epoch', 'N/A')}, val_loss={checkpoint.get('val_loss', 'N/A'):.6f}")

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    total_mse = 0.0
    total_mae = 0.0

    all_y = []
    all_y_hat = []

    logger.info(f"Starting evaluation loop on {len(test_ds)} test samples...")
    # 4. Evaluation Loop
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)

            total_mse += criterion_mse(y_hat, y).item() * x.size(0)
            total_mae += criterion_mae(y_hat, y).item() * x.size(0)

            # Save a few samples for plotting
            if plot_results and i == 0:
                all_y.append(y[0].cpu().numpy())
                all_y_hat.append(y_hat[0].cpu().numpy())

    avg_mse = total_mse / len(test_ds)
    avg_mae = total_mae / len(test_ds)

    metrics = {
        "mse": avg_mse,
        "rmse": np.sqrt(avg_mse),
        "mae": avg_mae
    }

    logger.info("=" * 50)
    logger.info(f"Evaluation Results for {checkpoint_path}:")
    for k, v in metrics.items():
        logger.info(f"  {k.upper()}: {v:.6f}")
    logger.info("=" * 50)
    logger.success("Evaluation complete!")

    if plot_results and all_y:
        logger.info("Generating evaluation plot...")
        plt.figure(figsize=(12, 4))
        plt.plot(all_y[0].flatten(), label="Appliance (Target)", alpha=0.7)
        plt.plot(all_y_hat[0].flatten(), label="Appliance (Predicted)", linestyle="--")
        plt.title("Energy Disaggregation: Appliance Power Prediction")
        plt.xlabel("Time samples")
        plt.ylabel("Power (normalized)")
        plt.legend()
        plt.grid(True)
        plt.savefig("evaluation_plot.png")
        logger.success("Plot saved to evaluation_plot.png")

    return metrics

if __name__ == "__main__":
    evaluate()
