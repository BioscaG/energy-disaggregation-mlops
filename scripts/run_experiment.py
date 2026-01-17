#!/usr/bin/env python
"""
Experiment runner: Execute training experiments using Hydra configuration management.

Hydra provides:
- Structured config validation (Pydantic)
- Automatic CLI parameter overrides
- Output directory management with timestamps
- Config composition and inheritance
- Parameter search capabilities

Usage:
  python run_experiment.py --config-path=../configs --config-name=quick_test
  python run_experiment.py --config-path=../configs --config-name=normal_training
  python run_experiment.py --config-path=../configs --config-name=normal_training \
    train.lr=0.001 train.epochs=100
"""

import subprocess
import sys
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


def run_command(cmd: list, description: str) -> bool:
    """Run a shell command and handle errors."""
    print(f"\n{'='*70}")
    print(f"▶️  {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        return False
    print(f"✅ {description} completed successfully")
    return True


def run_experiment(cfg: DictConfig, output_folder: Path) -> None:
    """Run complete experiment pipeline from Hydra config.

    Args:
        cfg: Hydra DictConfig with experiment parameters
        output_folder: Directory for storing outputs
    """
    # Generate processed data folder
    processed_folder = f"data/processed_{output_folder.name}"

    print(f"\n{'='*70}")
    print(f"📊 EXPERIMENT CONFIGURATION")
    print(f"{'='*70}")
    print(OmegaConf.to_yaml(cfg))
    print(f"Output folder: {output_folder}\n")

    # Step 1: Preprocess
    preprocess_cmd = [
        "python",
        "-m",
        "energy_dissagregation_mlops.cli",
        "preprocess",
        "--data-path",
        str(cfg.data_path),
        "--output-folder",
        processed_folder,
        "--building",
        str(cfg.preprocess.building),
        "--meter-mains",
        str(cfg.preprocess.meter_mains),
        "--meter-appliance",
        str(cfg.preprocess.meter_appliance),
        "--window-size",
        str(cfg.preprocess.window_size),
        "--stride",
        str(cfg.preprocess.stride),
        "--resample-rule",
        str(cfg.preprocess.resample_rule),
    ]

    if cfg.preprocess.max_samples:
        preprocess_cmd.extend(
            ["--max-samples", str(cfg.preprocess.max_samples)]
        )

    if not run_command(preprocess_cmd, "Preprocessing data"):
        return

    # Step 2: Train
    train_cmd = [
        "python",
        "-m",
        "energy_dissagregation_mlops.cli",
        "train",
        "--preprocessed-folder",
        processed_folder,
        "--epochs",
        str(cfg.train.epochs),
        "--batch-size",
        str(cfg.train.batch_size),
        "--lr",
        str(cfg.train.lr),
        "--num-workers",
        str(cfg.train.num_workers),
        "--device",
        str(cfg.train.device),
    ]

    if not run_command(train_cmd, "Training model"):
        return

    # Step 3: Evaluate
    evaluate_cmd = [
        "python",
        "-m",
        "energy_dissagregation_mlops.cli",
        "evaluate",
        "--preprocessed-folder",
        processed_folder,
        "--batch-size",
        str(cfg.evaluate.batch_size),
        "--device",
        str(cfg.evaluate.device),
    ]

    if cfg.evaluate.plot_results:
        evaluate_cmd.append("--plot-results")

    if not run_command(evaluate_cmd, "Evaluating model"):
        return

    # Summary
    print(f"\n{'='*70}")
    print(f"✅ EXPERIMENT COMPLETED: {output_folder.name}")
    print(f"{'='*70}")
    print(f"Results saved to:")
    print(f"  - Preprocessed data: {processed_folder}/")
    print(f"  - Model checkpoint: models/best.pt")
    print(f"  - Evaluation plot: evaluation_plot.png")
    print(f"  - Hydra outputs: {output_folder}/")
    print()


def main():
    """Main entry point with Hydra initialization."""
    # Get absolute path to configs directory
    config_dir = Path(__file__).parent.parent / "configs"
    config_dir = config_dir.resolve()

    # Verify config directory exists
    if not config_dir.exists():
        print(f"❌ Config directory not found: {config_dir}")
        sys.exit(1)

    # Initialize Hydra with config directory
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None, config_dir=str(config_dir)
    ):
        # Use compose to load config with CLI overrides
        # Default config name can be overridden via --config-name
        cfg = compose(config_name="normal_training")

        # Verify data exists
        if not Path(cfg.data_path).exists():
            print(f"❌ Dataset not found: {cfg.data_path}")
            print("   Run: python -m energy_dissagregation_mlops.cli download")
            sys.exit(1)

        # Get output folder from Hydra (includes timestamp)
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        try:
            run_experiment(cfg, output_dir)
        except KeyboardInterrupt:
            print("\n⚠️  Experiment interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
