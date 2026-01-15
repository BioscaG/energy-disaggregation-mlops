
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

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
    print(f"Evaluating on {device}...")

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

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = Model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    
    total_mse = 0.0
    total_mae = 0.0
    
    all_y = []
    all_y_hat = []

    # 4. Evaluation Loop
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            x = x.to(device)
            
            y = x
            
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

    print("-" * 30)
    print(f"Results for {checkpoint_path}:")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.6f}")
    print("-" * 30)

    if plot_results and all_y:
        plt.figure(figsize=(12, 4))
        plt.plot(all_y[0].flatten(), label="Original (Mains)", alpha=0.7)
        plt.plot(all_y_hat[0].flatten(), label="Reconstructed", linestyle="--")
        plt.title("Signal Reconstruction Evaluation")
        plt.legend()
        plt.grid(True)
        plt.savefig("evaluation_plot.png")
        print("✅ Plot saved to evaluation_plot.png")

    return metrics

if __name__ == "__main__":
    evaluate()