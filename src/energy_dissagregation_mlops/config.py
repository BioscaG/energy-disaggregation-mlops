"""Configuration classes for Hydra-based experiment management."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing."""

    building: int = 1
    meter_mains: int = 1
    meter_appliance: int = 2
    max_samples: Optional[int] = 360000
    window_size: int = 1024
    stride: int = 256
    resample_rule: str = "6S"
    normalize: bool = True
    clip_min: float = 0.0
    clip_max: Optional[float] = None


@dataclass
class TrainConfig:
    """Configuration for model training."""

    epochs: int = 50
    batch_size: int = 32
    lr: float = 0.0001
    num_workers: int = 4
    device: str = "auto"


@dataclass
class EvaluateConfig:
    """Configuration for model evaluation."""

    batch_size: int = 32
    device: str = "auto"
    plot_results: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    experiment_name: str = "default_experiment"
    data_path: str = "data/raw/ukdale.h5"
