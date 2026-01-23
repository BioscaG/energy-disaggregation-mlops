# Training Guide

## Overview

This guide covers training the energy disaggregation model from scratch, including configuration, monitoring, and hyperparameter tuning.

## Prerequisites

```bash
# Install dependencies
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_dev.txt

# Download dataset
python scripts/download_dataset.py
```

**Storage needed**: ~2 GB for UK-DALE dataset

## Training Process

### Quick Test (5 epochs, 2 minutes)

```bash
python scripts/run_experiment.py --config-name quick_test
```

**Output**:
- Model checkpoint: `models/last.pt`
- W&B logs: `wandb/run-*/`
- Metrics: Logged to console and W&B

### Normal Training (50 epochs, 30 minutes)

```bash
python scripts/run_experiment.py --config-name normal_training
```

**Best practices**:
- Run on GPU for 5-10x speedup
- Monitor W&B dashboard in real-time
- Check for overfitting via val_loss curve

### Advanced Training (Full Dataset)

```bash
python scripts/run_experiment.py --config-name full_training
```

**Expected results**:
- Train loss: ~0.08
- Val loss: ~0.12
- MAE: ~0.15 kW
- R²: ~0.85

## Configuration

### Hydra YAML Structure

Configuration files in `configs/` use Hydra framework:

```yaml
# configs/normal_training.yaml
data:
  path: data/preprocessed
  batch_size: 64
  val_split: 0.2

model:
  type: tcn
  hidden_dim: 128
  kernel_size: 7
  layers: 3

train:
  epochs: 50
  learning_rate: 0.001
  optimizer: adam
  device: cuda
  seed: 42

logging:
  wandb: true
  wandb_project: energy-disaggregation
```

### Creating Custom Config

```bash
# Copy and modify existing config
cp configs/normal_training.yaml configs/my_experiment.yaml

# Edit my_experiment.yaml
nano configs/my_experiment.yaml

# Run with custom config
python scripts/run_experiment.py --config-name my_experiment
```

### Command Line Overrides

Override any config parameter:

```bash
# Increase learning rate
python scripts/run_experiment.py --config-name normal_training train.learning_rate=0.01

# Use CPU instead of GPU
python scripts/run_experiment.py --config-name normal_training train.device=cpu

# Multiple overrides
python scripts/run_experiment.py --config-name normal_training \
  train.epochs=100 \
  train.learning_rate=0.01 \
  model.hidden_dim=256
```

## Monitoring with Weights & Biases

### Setup W&B

```bash
# Login to W&B account
wandb login

# Create project
wandb online  # Enable internet connection
```

### During Training

1. **W&B Dashboard**: https://wandb.ai/your-username/energy-disaggregation
2. **Real-time metrics**:
   - Training loss
   - Validation loss
   - Learning rate
   - GPU memory usage
   - Training time

3. **Artifact storage**:
   - Model checkpoints
   - Configuration files
   - Plots and visualizations

### Example Dashboard

```
Epoch 1/50  Loss: 0.234  Val Loss: 0.245
Epoch 2/50  Loss: 0.198  Val Loss: 0.210
...
Epoch 50/50  Loss: 0.082  Val Loss: 0.121

Best model saved at epoch 42
```

## Hyperparameter Tuning

### Grid Search

```bash
python scripts/run_sweep.py --config-name wandb_sweep
```

**Sweep configuration** (`configs/wandb_sweep.yaml`):
```yaml
method: grid
parameters:
  train.learning_rate:
    values: [0.0001, 0.001, 0.01]
  model.hidden_dim:
    values: [64, 128, 256]
  train.batch_size:
    values: [32, 64, 128]
```

### Random Search

```yaml
method: random
parameters:
  train.learning_rate:
    distribution: log_uniform
    min: 0.00001
    max: 0.01
```

### Bayesian Optimization

```yaml
method: bayes
parameters:
  train.learning_rate:
    distribution: log_uniform
    min: 0.00001
    max: 0.01
```

### Tracking Sweep Results

```bash
# View all runs in sweep
wandb sweep results --project energy-disaggregation

# Compare best models
wandb compare --project energy-disaggregation run1 run2 run3
```

## Performance Profiling

### Profile Training Loop

```bash
python scripts/profile_training.py --config-name normal_training
```

**Output**: `profiling_results/training_profile.json`

**Breakdown**:
- Data loading: ~15%
- Forward pass: ~40%
- Backward pass: ~35%
- Optimization step: ~10%

### Identify Bottlenecks

```bash
# CPU profiling
python -m cProfile -s cumulative scripts/run_experiment.py \
  --config-name quick_test

# PyTorch profiler
# (Already integrated in train.py with --profile flag)
python scripts/run_experiment.py --config-name quick_test --profile
```

## Model Checkpointing

### Save Best Model

Automatically saved during training:
```
models/
├── best.pt      # Best validation loss
├── last.pt      # Last epoch
└── checkpoint_epoch_25.pt
```

### Load Checkpoint

```python
from src.energy_dissagregation_mlops.model import TemporalConvNet
import torch

# Load best model
model = TemporalConvNet(input_size=1, num_channels=[128, 128, 128])
model.load_state_dict(torch.load("models/best.pt"))
model.eval()

# Make predictions
with torch.no_grad():
    output = model(torch.randn(1, 1, 100))
```

### Resume Training

```bash
# Continue from checkpoint
python scripts/run_experiment.py \
  --config-name normal_training \
  train.checkpoint="models/checkpoint_epoch_25.pt"
```

## Advanced Topics

### Mixed Precision Training

Enable for faster training on V100/A100 GPUs:

```yaml
train:
  amp: true  # Automatic Mixed Precision
```

**Benefits**:
- 30-50% faster
- ~50% less memory
- No accuracy loss with proper tuning

### Distributed Training

Multi-GPU training:

```bash
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  scripts/run_experiment.py --config-name normal_training
```

### Data Augmentation

```yaml
data:
  augment: true
  augmentation:
    - gaussian_noise: 0.01
    - time_shift: 5
    - time_stretch: 0.9-1.1
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python scripts/run_experiment.py --config-name normal_training \
  train.batch_size=32

# Or use CPU
python scripts/run_experiment.py --config-name normal_training \
  train.device=cpu
```

### Model Not Improving

1. **Check learning rate**: Try values 10x higher/lower
2. **Verify data**: Inspect sample batches with `visualize.py`
3. **Increase model capacity**: Increase `hidden_dim` or `layers`
4. **More data**: Verify data loading works correctly

### Training Crashes

```bash
# Check logs
tail -100 ~/.cache/energy_disaggregation_mlops/train.log

# Run with verbose output
python scripts/run_experiment.py --config-name quick_test -vv
```

## Exporting Models

### Export to ONNX

```bash
python scripts/export_onnx.py --model models/best.pt --output models/model.onnx
```

**Verification**:
```bash
python -c "import onnx; model = onnx.load('models/model.onnx'); print(onnx.checker.check_model(model))"
```

### Export to TorchScript

```python
import torch
from src.energy_dissagregation_mlops.model import TemporalConvNet

model = TemporalConvNet(1, [128, 128, 128])
model.load_state_dict(torch.load("models/best.pt"))

scripted = torch.jit.script(model)
torch.jit.save(scripted, "models/model.jit")
```

## Next Steps

- Deploy trained model via [Deployment Guide](deployment.md)
- Monitor in production via [Drift Detection](drift-detection.md)
- Analyze results via [W&B Reports](https://wandb.ai)
