# Getting Started

## Installation

### Prerequisites

- Python 3.12 or higher
- Git
- (Optional) CUDA 11.8+ for GPU support
- (Optional) Docker 20.10+

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/energy-disaggregation-mlops.git
cd energy-disaggregation-mlops
```

### Step 2: Create Virtual Environment

```bash
# Using venv (recommended)
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install runtime and development dependencies
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

## Verification

Verify the installation is correct:

```bash
# Test imports
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "from energy_dissagregation_mlops.model import Model; print('✓ Model import successful')"

# Check pre-commit hooks are installed
pre-commit install
pre-commit run --all-files
```

## First Training Run

### Quick Test (5 minutes)

Perfect for verifying everything works:

```bash
python scripts/run_experiment.py --config-name quick_test
```

This runs:
- 5 epochs of training
- Full preprocessing pipeline
- Validation every epoch
- Model checkpoint saving
- Metrics logged to W&B

Expected output:
```
INFO:root:Starting training with device=cpu, epochs=5, batch_size=32, lr=0.001
INFO:root:Dataset loaded: 1000 samples
Epoch 1/5: train_loss=0.256, val_loss=0.198
Epoch 2/5: train_loss=0.184, val_loss=0.156
...
INFO:root:Model saved to: models/best.pt
```

### Full Training (2-3 hours)

For complete model training:

```bash
python scripts/run_experiment.py --config-name normal_training
```

This runs:
- 50 epochs of training
- Full dataset
- More aggressive hyperparameters
- Complete metrics logging

### Custom Configuration

Override hyperparameters from CLI:

```bash
python scripts/run_experiment.py \
  --config-name normal_training \
  train.lr=0.0001 \
  train.epochs=100 \
  train.batch_size=64 \
  train.device=cuda
```

## Running the API

### Local Deployment

```bash
# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, test it
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "timestamp": "2026-01-23T12:00:00"
}
```

### Make Predictions

#### PyTorch Backend

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  }'
```

#### ONNX Backend (faster)

```bash
curl -X POST http://localhost:8000/predict/onnx \
  -H "Content-Type: application/json" \
  -d '{
    "x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  }'
```

Response:
```json
{
  "batch_size": 1,
  "t": 10,
  "y": [0.245, 0.318, 0.412, 0.521, 0.643, 0.771, 0.901, 0.023, 0.145, 0.267]
}
```

### Interactive Web Frontend

Open in browser:
```
http://127.0.0.1:5501/frontend/index.html
```

Features:
- Real-time API testing
- Compare PyTorch vs ONNX predictions
- Visualize results
- Monitor API health

## Using Docker

### Build API Container

```bash
docker build -f dockerfiles/api.dockerfile -t energy-api:latest .
```

### Run API Container

```bash
docker run \
  -p 8000:8000 \
  -v ./models:/app/models \
  -e MODEL_PATH=/app/models/best.pt \
  energy-api:latest
```

### Build Training Container

```bash
docker build -f dockerfiles/train.dockerfile -t energy-train:latest .
```

### Run Training with GPU

```bash
docker run \
  --gpus all \
  -v ./data:/app/data \
  -v ./models:/app/models \
  -v ./configs:/app/configs \
  energy-train:latest \
  python scripts/run_experiment.py --config-name quick_test
```

## Understanding Configuration Files

Hydra configs are in `configs/` directory:

```yaml
# configs/quick_test.yaml
train:
  epochs: 5
  batch_size: 32
  lr: 0.001
  device: cpu

preprocessing:
  window_size: 256
  stride: 256
  resample_freq: 60  # 1 minute
```

To use:
```bash
python scripts/run_experiment.py --config-name quick_test
```

To override:
```bash
python scripts/run_experiment.py --config-name quick_test train.epochs=10 train.lr=0.0005
```

## Monitoring Training

### Weights & Biases

Access your training runs at: https://wandb.ai/your-username/energy-disaggregation

Features:
- Real-time metric graphs
- Hyperparameter comparison
- Model artifact storage
- Experiment history

Initialize W&B:
```bash
wandb login
# Enter your API key from https://wandb.ai/authorize
```

### Local Logs

Loguru saves detailed logs to `logs/` directory:

```bash
# View latest logs
tail -f logs/*.log

# Search for errors
grep ERROR logs/*.log
```

## Common Issues

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU instead
python scripts/run_experiment.py --config-name quick_test train.device=cpu
```

### Out of Memory

```bash
# Reduce batch size
python scripts/run_experiment.py --config-name quick_test train.batch_size=16

# Reduce sequence length
python scripts/run_experiment.py --config-name quick_test preprocessing.window_size=128
```

### Data Not Found

```bash
# Download and preprocess data
python -m energy_dissagregation_mlops.cli preprocess \
  --data-path data/raw/ukdale.h5 \
  --output-folder data/processed
```

### Pre-commit Hooks Failing

```bash
# See what hooks are checking
pre-commit run --all-files -v

# Fix formatting automatically
ruff format .
ruff check --fix .

# Retry commit
git commit -m "Your message"
```

## Next Steps

- [Read CLI Usage Guide](cli-usage.md) for more commands
- [Check API Reference](api-reference.md) for endpoint details
- [Review Training Guide](training.md) for advanced configuration
- [Explore Deployment Options](deployment.md) for production setup

## Getting Help

- Check [troubleshooting](../DOCKER.md) section
- Review [GitHub Issues](https://github.com/yourusername/energy-disaggregation-mlops/issues)
- Check CI/CD logs in GitHub Actions
- Review Loguru logs in `logs/` directory

---

**Questions?** Open an issue on GitHub!
