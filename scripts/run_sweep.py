#!/usr/bin/env python3
"""
Hyperparameter sweep script using Weights & Biases.
Provides a programmatic way to run sweeps without YAML.
"""

import wandb
from energy_dissagregation_mlops.train import train


def sweep_train():
    """Training function for W&B sweep."""
    # Initialize W&B run (sweep agent does this automatically)
    wandb.init()

    # Get hyperparameters from sweep config
    config = wandb.config

    # Run training with sweep hyperparameters
    train(
        preprocessed_folder=config.get("preprocessed_folder", "data/preprocessed"),
        batch_size=config.get("batch_size", 32),
        lr=config.get("lr", 1e-4),
        epochs=config.get("epochs", 50),
        num_workers=config.get("num_workers", 2),
        device=config.get("device", None),
        use_wandb=True,
        project_name=config.get("project_name", "energy-disaggregation-sweep"),
    )


def create_sweep():
    """Create a sweep configuration and return sweep ID."""
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "batch_size": {"values": [16, 32, 64]},
            "lr": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1e-2,
            },
            "epochs": {"value": 50},
            "preprocessed_folder": {"value": "data/preprocessed"},
            "num_workers": {"value": 2},
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 10,
            "eta": 3,
            "s": 2,
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="energy-disaggregation-sweep")

    print(f"Created sweep with ID: {sweep_id}")
    print(f"Run sweep with: wandb agent {sweep_id}")

    return sweep_id


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "create":
        # Create sweep and print ID
        create_sweep()
    else:
        # Run sweep training (called by wandb agent)
        sweep_train()
