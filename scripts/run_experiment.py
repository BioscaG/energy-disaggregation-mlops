#!/usr/bin/env python
"""
Experiment runner: Execute training experiments from config files.

Usage:
  python run_experiment.py configs/quick_test.yaml
  python run_experiment.py configs/normal_training.yaml --output-name my_experiment
"""

import argparse
import yaml
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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


def run_experiment(config_path: str, output_name: str = None) -> None:
    """Run complete experiment pipeline from config."""

    # Load config
    print(f"📂 Loading config from: {config_path}")
    config = load_config(config_path)

    # Generate output folder name
    if not output_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = Path(config_path).stem
        output_name = f"{config_name}_{timestamp}"

    processed_folder = f"data/processed_{output_name}"

    print(f"📋 Config loaded:")
    print(f"  Preprocess: meter_appliance={config['preprocess']['meter_appliance']}, "
          f"max_samples={config['preprocess']['max_samples']}")
    print(f"  Train: epochs={config['train']['epochs']}, lr={config['train']['lr']}")
    print(f"  Output folder: {processed_folder}")

    # Step 1: Preprocess
    preprocess_cmd = [
        "python", "-m", "energy_dissagregation_mlops.cli", "preprocess",
        "--data-path", "data/raw/ukdale.h5",
        "--output-folder", processed_folder,
        "--building", str(config['preprocess']['building']),
        "--meter-mains", str(config['preprocess']['meter_mains']),
        "--meter-appliance", str(config['preprocess']['meter_appliance']),
        "--window-size", str(config['preprocess']['window_size']),
        "--stride", str(config['preprocess']['stride']),
        "--resample-rule", str(config['preprocess']['resample_rule']),
    ]

    if config['preprocess']['max_samples']:
        preprocess_cmd.extend(["--max-samples", str(config['preprocess']['max_samples'])])

    if not run_command(preprocess_cmd, "Preprocessing data"):
        return

    # Step 2: Train
    train_cmd = [
        "python", "-m", "energy_dissagregation_mlops.cli", "train",
        "--preprocessed-folder", processed_folder,
        "--epochs", str(config['train']['epochs']),
        "--batch-size", str(config['train']['batch_size']),
        "--lr", str(config['train']['lr']),
        "--num-workers", str(config['train']['num_workers']),
        "--device", str(config['train']['device']),
    ]

    if not run_command(train_cmd, "Training model"):
        return

    # Step 3: Evaluate
    evaluate_cmd = [
        "python", "-m", "energy_dissagregation_mlops.cli", "evaluate",
        "--preprocessed-folder", processed_folder,
        "--batch-size", str(config['evaluate']['batch_size']),
        "--device", str(config['evaluate']['device']),
    ]

    if config['evaluate']['plot_results']:
        evaluate_cmd.append("--plot-results")

    if not run_command(evaluate_cmd, "Evaluating model"):
        return

    # Summary
    print(f"\n{'='*70}")
    print(f"✅ EXPERIMENT COMPLETED: {output_name}")
    print(f"{'='*70}")
    print(f"Results saved to:")
    print(f"  - Preprocessed data: {processed_folder}/")
    print(f"  - Model checkpoint: models/best.pt")
    print(f"  - Evaluation plot: evaluation_plot.png")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run energy disaggregation experiments from config files"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to config YAML file (e.g., configs/quick_test.yaml)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom name for output folders (default: config_name_timestamp)"
    )

    args = parser.parse_args()

    # Verify config exists
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    # Verify data exists
    if not Path("data/raw/ukdale.h5").exists():
        print(f"❌ Dataset not found: data/raw/ukdale.h5")
        print("   Run: python -m energy_dissagregation_mlops.cli download")
        sys.exit(1)

    try:
        run_experiment(args.config, args.output_name)
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
