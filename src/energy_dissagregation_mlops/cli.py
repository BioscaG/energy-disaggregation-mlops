from pathlib import Path

import numpy as np
import torch
import typer
from loguru import logger

from energy_dissagregation_mlops.data import MyDataset, PreprocessConfig, download_ukdale
from energy_dissagregation_mlops.drift_detection import (
    DataDriftDetector,
    RobustnessAnalyzer,
    visualize_drift_analysis,
    visualize_robustness_analysis,
)
from energy_dissagregation_mlops.evaluate import evaluate as evaluate_fn
from energy_dissagregation_mlops.model import Model
from energy_dissagregation_mlops.train import train as train_fn

app = typer.Typer(no_args_is_help=True)


@app.command()
def preprocess(
    data_path: Path = typer.Option(..., exists=True, help="Path to ukdale.h5"),
    output_folder: Path = typer.Option(..., help="Folder to write processed chunks"),
    building: int = typer.Option(1, help="UK-DALE building number (house)"),
    meter_mains: int = typer.Option(1, help="Meter number for mains/total power (usually 1)"),
    meter_appliance: int = typer.Option(2, help="Meter number for appliance to predict (2+)"),
    window_size: int = typer.Option(1024, help="Window length in samples"),
    stride: int = typer.Option(256, help="Stride between windows"),
    resample_rule: str = typer.Option("6S", help="Pandas resample rule (e.g. 6S, 1min)"),
    power_type: str = typer.Option("apparent", help="Power type: apparent/active if available"),
    normalize: bool = typer.Option(True, help="Z-score normalize using global mean/std"),
    max_samples: int = typer.Option(None, help="Limit to first N samples for faster testing (None=all)"),
):
    logger.info("CLI: Starting preprocessing command")
    cfg = PreprocessConfig(
        building=building,
        meter_mains=meter_mains,
        meter_appliance=meter_appliance,
        physical_quantity="power",
        power_type=power_type,
        resample_rule=resample_rule if resample_rule.lower() != "none" else None,
        window_size=window_size,
        stride=stride,
        normalize=normalize,
        max_samples=max_samples,
    )
    ds = MyDataset(data_path=data_path)
    ds.preprocess(output_folder=output_folder, cfg=cfg)
    logger.success(f"Preprocessing complete! Saved to {output_folder}")
    logger.info(
        f"Configuration: mains={meter_mains}, appliance={meter_appliance}, window={window_size}, stride={stride}"
    )
    if max_samples:
        logger.info(f"Limited to {max_samples} samples")


@app.command()
def train(
    preprocessed_folder: Path = typer.Option("data/processed", exists=True, help="Folder with chunk_*.npz + meta.npz"),
    epochs: int = typer.Option(100, help="Epochs"),
    batch_size: int = typer.Option(16, help="Batch size"),
    lr: float = typer.Option(1e-4, help="Learning rate"),
    num_workers: int = typer.Option(2, help="DataLoader workers"),
    device: str = typer.Option("auto", help="auto/cpu/cuda"),
    use_wandb: bool = typer.Option(True, help="Enable Weights & Biases logging"),
    wandb_project: str = typer.Option("energy-disaggregation", help="W&B project name"),
    run_name: str = typer.Option(None, help="W&B run name"),
):
    logger.info("CLI: Starting training command")
    train_fn(
        preprocessed_folder=str(preprocessed_folder),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        device=None if device == "auto" else device,
        use_wandb=use_wandb,
        project_name=wandb_project,
        run_name=run_name,
    )


@app.command()
def evaluate(
    preprocessed_folder: Path = typer.Option("data/processed", exists=True, help="Folder with processed data"),
    checkpoint_path: Path = typer.Option("models/best.pt", exists=True, help="Path to model checkpoint"),
    batch_size: int = typer.Option(32, help="Batch size"),
    device: str = typer.Option("auto", help="auto/cpu/cuda"),
    plot_results: bool = typer.Option(False, help="Save a reconstruction plot"),
):
    logger.info("CLI: Starting evaluation command")
    evaluate_fn(
        preprocessed_folder=str(preprocessed_folder),
        checkpoint_path=str(checkpoint_path),
        batch_size=batch_size,
        device=None if device == "auto" else device,
        plot_results=plot_results,
    )


@app.command()
def download(
    target_dir: Path = typer.Option("data/raw", help="Where to store raw dataset"),
):
    logger.info("CLI: Starting download command")
    download_ukdale(target_dir)
    logger.success(f"Dataset downloaded to {target_dir}")


@app.command()
def detect_drift(
    preprocessed_folder: Path = typer.Option("data/processed", exists=True, help="Folder with processed data"),
    data_path: Path = typer.Option("data/raw/ukdale.h5", exists=True, help="Path to raw dataset"),
    test_split_ratio: float = typer.Option(0.1, help="Portion of data to use for drift test"),
    drift_type: str = typer.Option("scale", help="Type of drift to simulate: scale/noise/missing"),
    drift_magnitude: float = typer.Option(0.2, help="Magnitude of drift (e.g., 0.2 = 20% shift)"),
):
    """
    Detect data drift in the dataset.

    Simulates different types of distribution shifts and measures them using:
    - Kolmogorov-Smirnov test
    - Population Stability Index (PSI)
    - Maximum Mean Discrepancy (MMD)
    """
    logger.info("CLI: Starting drift detection command")

    # Load dataset
    dataset = MyDataset(data_path=data_path, preprocessed_folder=preprocessed_folder)
    n = len(dataset)
    n_test = int(n * test_split_ratio)

    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [n - n_test, n_test], generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Loaded {len(train_ds)} training and {len(test_ds)} test samples")

    # Extract mains power from both sets
    train_mains = []
    test_mains = []

    for idx in range(min(len(train_ds), 100)):  # Sample for efficiency
        train_mains.append(train_ds[idx][0].numpy().flatten())

    for idx in range(min(len(test_ds), 100)):
        test_mains.append(test_ds[idx][0].numpy().flatten())

    train_mains = np.concatenate(train_mains)
    test_mains = np.concatenate(test_mains)

    logger.info(f"Train mains power - mean: {train_mains.mean():.2f}, std: {train_mains.std():.2f}")

    # Apply drift to test data
    if drift_type == "scale":
        drifted_mains = test_mains * (1 + drift_magnitude)
        logger.info(f"Applied {drift_magnitude * 100:.1f}% scale drift")
    elif drift_type == "noise":
        noise = np.random.normal(0, drift_magnitude, test_mains.shape)
        drifted_mains = test_mains + noise
        logger.info(f"Added Gaussian noise (std={drift_magnitude:.3f})")
    elif drift_type == "missing":
        drifted_mains = test_mains.copy()
        num_missing = int(len(drifted_mains) * drift_magnitude)
        missing_idx = np.random.choice(len(drifted_mains), num_missing, replace=False)
        drifted_mains[missing_idx] = 0
        logger.info(f"Applied {drift_magnitude * 100:.1f}% missing data")
    else:
        raise ValueError(f"Unknown drift type: {drift_type}")

    logger.info(f"Test (drifted) mains power - mean: {drifted_mains.mean():.2f}, std: {drifted_mains.std():.2f}")

    # Detect drift
    logger.info("\n" + "=" * 60)
    logger.info("DRIFT DETECTION RESULTS")
    logger.info("=" * 60)

    result = DataDriftDetector.compare_distributions(train_mains, drifted_mains, "mains_power")

    logger.info(f"Kolmogorov-Smirnov Test:")
    logger.info(f"  Statistic: {result['ks_test']['statistic']:.4f}")
    logger.info(f"  P-value: {result['ks_test']['p_value']:.6f}")
    logger.info(f"  Drift Detected: {result['ks_test']['drift_detected']}")

    logger.info(f"\nPopulation Stability Index (PSI):")
    logger.info(f"  PSI: {result['psi']:.4f}")
    logger.info(f"  Threshold: 0.25 (significant drift)")
    logger.info(f"  Significant: {result['psi'] > 0.25}")

    logger.info(f"\nMaximum Mean Discrepancy (MMD):")
    logger.info(f"  MMD: {result['mmd']:.4f}")

    logger.info(f"\nDistribution Statistics:")
    logger.info(f"  Reference Mean: {result['reference_mean']:.4f}")
    logger.info(f"  Current Mean: {result['current_mean']:.4f}")
    logger.info(f"  Mean Shift: {result['mean_shift']:.4f}")
    logger.info("=" * 60 + "\n")

    # Save visualization
    visualize_drift_analysis(result, output_path="drift_analysis_report.png")
    logger.success("Drift analysis visualization saved to drift_analysis_report.png")


@app.command()
def test_robustness(
    preprocessed_folder: Path = typer.Option("data/processed", exists=True, help="Folder with processed data"),
    data_path: Path = typer.Option("data/raw/ukdale.h5", exists=True, help="Path to raw dataset"),
    checkpoint_path: Path = typer.Option("models/best.pt", exists=True, help="Path to model checkpoint"),
    perturbation_type: str = typer.Option("noise", help="Type: noise/scale/missing"),
    num_samples: int = typer.Option(50, help="Number of samples to test"),
    batch_size: int = typer.Option(32, help="Batch size"),
):
    """
    Test model robustness against various perturbations.

    Evaluates how well the model performs under:
    - Gaussian noise injection
    - Amplitude scaling (sensor drift)
    - Missing data simulation
    """
    logger.info("CLI: Starting robustness testing command")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    model = Model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    logger.success("Model loaded")

    # Load test data
    dataset = MyDataset(data_path=data_path, preprocessed_folder=preprocessed_folder)
    n = len(dataset)
    n_train = int(0.9 * n)
    _, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n - n_train], generator=torch.Generator().manual_seed(42)
    )

    # Sample data
    from torch.utils.data import Subset

    indices = np.random.choice(len(test_ds), min(num_samples, len(test_ds)), replace=False)
    test_ds_sampled = Subset(test_ds, indices)

    loader = torch.utils.data.DataLoader(test_ds_sampled, batch_size=batch_size, shuffle=False)
    x_test_list, y_test_list = [], []
    for x, y in loader:
        x_test_list.append(x)
        y_test_list.append(y)

    x_test = torch.cat(x_test_list).to(device)
    y_test = torch.cat(y_test_list).to(device)

    logger.info(f"Testing robustness on {len(x_test)} samples")

    # Test robustness
    analyzer = RobustnessAnalyzer(model, device=device)

    if perturbation_type == "noise":
        levels = [0.0, 0.01, 0.05, 0.1, 0.15]
    elif perturbation_type == "scale":
        levels = [0.6, 0.8, 1.0, 1.2, 1.5]
    elif perturbation_type == "missing":
        levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")

    logger.info(f"\nTesting robustness to {perturbation_type} perturbations...")
    results = analyzer.evaluate_robustness(
        x_test, y_test, perturbation_type=perturbation_type, perturbation_levels=levels, batch_size=batch_size
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"ROBUSTNESS TEST RESULTS ({perturbation_type})")
    logger.info("=" * 60)

    for level in sorted(results["metrics_by_level"].keys()):
        metrics = results["metrics_by_level"][level]
        logger.info(f"\nLevel {level}:")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        logger.info(f"  MAE:  {metrics['mae']:.6f}")
        logger.info(f"  MSE:  {metrics['mse']:.6f}")

    logger.info("=" * 60 + "\n")

    # Calculate degradation
    baseline_rmse = results["baseline_metric"]["rmse"]
    worst_case = max(results["metrics_by_level"].values(), key=lambda m: m["rmse"])
    degradation = (worst_case["rmse"] - baseline_rmse) / baseline_rmse * 100

    logger.info(f"Performance Degradation:")
    logger.info(f"  Baseline RMSE: {baseline_rmse:.6f}")
    logger.info(f"  Worst-case RMSE: {worst_case['rmse']:.6f}")
    logger.info(f"  Degradation: {degradation:.1f}%")

    if degradation < 20:
        logger.success("Model is ROBUST ✅ (degradation < 20%)")
    elif degradation < 50:
        logger.warning("Model has MODERATE robustness ⚠️ (degradation 20-50%)")
    else:
        logger.error("Model has LOW robustness ❌ (degradation > 50%)")

    # Save visualization
    visualize_robustness_analysis(results, output_path="robustness_report.png")
    logger.success("Robustness analysis visualization saved to robustness_report.png")


def main():
    logger.info("Starting Energy Disaggregation MLOps CLI")
    app()


if __name__ == "__main__":
    main()
