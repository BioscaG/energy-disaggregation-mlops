# CLI Usage

## Overview

Command-line interface for training, evaluation, and deployment of the energy disaggregation model.

## Installation

```bash
# Install package in editable mode
pip install -e .

# Verify installation
energy-disaggregation-mlops --version
```

## Commands

### Training

Train a model from scratch or resume from checkpoint.

```bash
# Quick test (5 epochs)
python scripts/run_experiment.py --config-name quick_test

# Normal training (50 epochs)
python scripts/run_experiment.py --config-name normal_training

# Full training (all data, 200+ epochs)
python scripts/run_experiment.py --config-name full_training
```

**Options**:
```bash
python scripts/run_experiment.py --help
```

**Common configurations**:
- `quick_test.yaml`: 5 epochs, debug mode
- `normal_training.yaml`: 50 epochs, standard setup
- `full_training.yaml`: 200+ epochs, production
- `profiling.yaml`: Profile training loop
- `wandb_sweep.yaml`: Hyperparameter sweep

**Override parameters**:
```bash
# Change learning rate
python scripts/run_experiment.py --config-name normal_training \
  train.learning_rate=0.01

# Use CPU instead of GPU
python scripts/run_experiment.py --config-name normal_training \
  train.device=cpu

# Multiple overrides
python scripts/run_experiment.py --config-name normal_training \
  train.epochs=100 \
  train.batch_size=128 \
  model.hidden_dim=256
```

### Export Model

Export trained model to ONNX format for optimization.

```bash
python scripts/export_onnx.py \
  --model models/best.pt \
  --output models/model.onnx
```

**Options**:
- `--model`: Path to PyTorch model (default: `models/best.pt`)
- `--output`: Output ONNX path (default: `models/model.onnx`)
- `--opset`: ONNX opset version (default: 12)

**Verify export**:
```bash
python -c "
import onnx
model = onnx.load('models/model.onnx')
print(onnx.checker.check_model(model))
print('✓ ONNX model is valid')
"
```

### Download Dataset

Download UK-DALE dataset for training.

```bash
python scripts/download_dataset.py
```

**Output**:
- `data/raw/ukdale.h5` (2GB)
- `data/preprocessed/chunk_*.npz` (500MB)
- `data/preprocessed/meta.npz` (10KB)

**Options**:
- `--output-dir`: Output directory (default: `data/`)
- `--chunk-size`: Chunk size in hours (default: 24)
- `--force-redownload`: Force redownload (default: False)

### Profile Training

Analyze training loop bottlenecks.

```bash
python scripts/profile_training.py --config-name quick_test
```

**Output**: `profiling_results/training_profile.json`

**Breakdown**:
- Data loading time
- Forward pass time
- Backward pass time
- Optimization step time

**Visualize**:
```python
import json
import matplotlib.pyplot as plt

with open('profiling_results/training_profile.json') as f:
    profile = json.load(f)

times = profile['timing_breakdown']
plt.bar(times.keys(), times.values())
plt.ylabel('Time (seconds)')
plt.title('Training Loop Breakdown')
plt.show()
```

### Save Reference Distribution

Save reference data distribution for drift detection.

```bash
python scripts/save_reference_distribution.py \
  --output models/reference_distribution.npy \
  --data-path data/preprocessed/
```

**Output**: `models/reference_distribution.npy` (7.7MB)

Used by drift detection system to identify distribution shifts in production.

### Run Hyperparameter Sweep

Run grid/random/Bayesian search for hyperparameters.

```bash
python scripts/run_sweep.py --config-name wandb_sweep
```

**Sweep configuration** (`configs/wandb_sweep.yaml`):
```yaml
method: grid                    # grid, random, bayes
parameters:
  train.learning_rate:
    values: [0.0001, 0.001, 0.01]
  model.hidden_dim:
    values: [64, 128, 256]
  train.batch_size:
    values: [32, 64, 128]
```

**Results**: All runs logged to W&B project

### API Server

Start FastAPI server for inference.

```bash
# Development (with auto-reload)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Production (with Gunicorn)
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

**Access**:
- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Test Drift Robustness

Test model robustness under drift scenarios.

```bash
python scripts/test_drift_robustness.py \
  --model models/best.pt \
  --data data/preprocessed/chunk_0000.npz \
  --output reports/drift_robustness.json
```

**Tests**:
- Gaussian noise robustness
- Time shift robustness
- Distribution shift robustness

**Output**: JSON report with robustness scores (0-1 scale)

### Example Drift Detection

Run example drift detection workflow.

```bash
python scripts/example_drift_detection.py \
  --reference models/reference_distribution.npy \
  --data data/preprocessed/chunk_0000.npz
```

**Output**:
```
KS Test: statistic=0.12, p_value=0.45
PSI Test: distance=0.08
MMD Test: statistic=0.34

✓ No drift detected
```

## Advanced Usage

### With Hydra

Hydra framework provides powerful configuration management.

```bash
# List all available configs
ls configs/

# Run with specific config
python scripts/run_experiment.py --config-name normal_training

# Override any parameter
python scripts/run_experiment.py --config-name normal_training \
  train.epochs=100 \
  train.learning_rate=0.01

# Multiple overrides on same group
python scripts/run_experiment.py --config-name normal_training \
  train.epochs=100 train.learning_rate=0.01 train.batch_size=128

# Sweep over values
python scripts/run_experiment.py --config-name normal_training \
  --multirun train.learning_rate=0.0001,0.001,0.01

# Show expanded config
python scripts/run_experiment.py --config-name normal_training --cfg all
```

### With Docker

Run any command in containerized environment.

```bash
# Build container
docker build -t energy-disaggregation-cli:latest \
  -f dockerfiles/cli.dockerfile .

# Run training
docker run -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  energy-disaggregation-cli:latest \
  python scripts/run_experiment.py --config-name quick_test

# Run with GPU
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  energy-disaggregation-cli:latest \
  python scripts/run_experiment.py --config-name normal_training
```

### With W&B Integration

Log experiments to Weights & Biases dashboard.

```bash
# Login to W&B
wandb login

# Run training with W&B logging
python scripts/run_experiment.py --config-name normal_training

# View results
# https://wandb.ai/username/energy-disaggregation
```

### Batch Processing

Process multiple experiments sequentially.

```bash
# Process 3 experiments
python scripts/run_experiment.py --config-name normal_training train.seed=1
python scripts/run_experiment.py --config-name normal_training train.seed=2
python scripts/run_experiment.py --config-name normal_training train.seed=3

# Or use multirun
python scripts/run_experiment.py --config-name normal_training \
  --multirun train.seed=1,2,3
```

## Environment Variables

Control behavior with environment variables.

```bash
# Logging level
LOGLEVEL=INFO python scripts/run_experiment.py --config-name quick_test

# W&B settings
WANDB_PROJECT=energy-disaggregation python scripts/run_experiment.py
WANDB_ENTITY=username python scripts/run_experiment.py
WANDB_MODE=offline python scripts/run_experiment.py  # No internet

# PyTorch settings
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_experiment.py  # Use GPUs 0,1
OMP_NUM_THREADS=8 python scripts/run_experiment.py         # CPU threads

# Data settings
DATA_PATH=/custom/data/path python scripts/run_experiment.py
MODEL_PATH=/custom/model/path python scripts/run_experiment.py
```

## Logging

### Log Levels

Control verbosity with log levels.

```bash
# Quiet (errors only)
python scripts/run_experiment.py --config-name quick_test -q

# Verbose (debug info)
python scripts/run_experiment.py --config-name quick_test -v

# Very verbose
python scripts/run_experiment.py --config-name quick_test -vv
```

### Log Files

Logs saved to: `~/.cache/energy_disaggregation_mlops/`

```bash
# View logs in real-time
tail -f ~/.cache/energy_disaggregation_mlops/train.log

# Search for errors
grep ERROR ~/.cache/energy_disaggregation_mlops/train.log
```

## Troubleshooting

### Command Not Found

If CLI commands not found:

```bash
# Reinstall package
pip install -e .

# Verify installation
which energy-disaggregation-mlops

# Check Python path
python -c "import src.energy_dissagregation_mlops; print(src.energy_dissagregation_mlops.__file__)"
```

### GPU Issues

If GPU not detected:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU explicitly
python scripts/run_experiment.py --config-name quick_test train.device=cpu

# Check CUDA version
nvidia-smi

# Set GPU visibility
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py
```

### Memory Issues

If out of memory:

```bash
# Reduce batch size
python scripts/run_experiment.py --config-name normal_training \
  train.batch_size=32

# Reduce model size
python scripts/run_experiment.py --config-name normal_training \
  model.hidden_dim=64

# Use CPU (slower but works)
python scripts/run_experiment.py --config-name quick_test train.device=cpu
```

### Data Issues

If data loading fails:

```bash
# Download dataset
python scripts/download_dataset.py

# Check data files
ls data/preprocessed/

# Verify data integrity
python -c "
import numpy as np
data = np.load('data/preprocessed/chunk_0000.npz')
print('Keys:', list(data.keys()))
print('X shape:', data['X'].shape)
"
```

## Performance Tips

1. **Use GPU**: 5-10x faster than CPU
2. **Reduce batch size**: If memory issues
3. **Profile training**: Identify bottlenecks
4. **Use ONNX**: For inference optimization
5. **Parallelize data loading**: Set `num_workers` in DataLoader

## Next Steps

- [Training Guide](training.md) for detailed training instructions
- [API Reference](api-reference.md) for API endpoints
- [Deployment Guide](deployment.md) for production setup
