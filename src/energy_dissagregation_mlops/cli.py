from pathlib import Path
import typer
from loguru import logger

from energy_dissagregation_mlops.data import MyDataset, PreprocessConfig, download_ukdale
from energy_dissagregation_mlops.train import train as train_fn
from energy_dissagregation_mlops.evaluate import evaluate as evaluate_fn

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
    logger.info(f"Configuration: mains={meter_mains}, appliance={meter_appliance}, window={window_size}, stride={stride}")
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

def main():
    logger.info("Starting Energy Disaggregation MLOps CLI")
    app()

if __name__ == "__main__":
    main()
