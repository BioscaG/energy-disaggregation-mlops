# Energy Disaggregation MLOps

Complete machine learning operations system for Non-Intrusive Load Monitoring (NILM) with production-ready deployment, monitoring, and optimization.

## What is NILM?

Non-Intrusive Load Monitoring breaks down household electricity consumption into individual appliances using only the aggregated mains signal—no smart plugs required.

## Quick Start

### Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt requirements_dev.txt
```

### Train Model
```bash
# Quick test (5 epochs)
python scripts/run_experiment.py --config-name quick_test

# Full training (50 epochs)
python scripts/run_experiment.py --config-name normal_training
```

### Run API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
```

### Docker
```bash
docker build -f dockerfiles/api.dockerfile -t energy-api .
docker run -p 8000:8000 -v ./models:/app/models energy-api
```

## Features

- ✅ **Production API**: FastAPI with /health and /predict endpoints
- ✅ **Model Optimization**: ONNX export (30-40% faster inference)
- ✅ **Experiment Tracking**: Weights & Biases integration
- ✅ **Drift Detection**: Kolmogorov-Smirnov test + PSI monitoring
- ✅ **CI/CD Pipelines**: GitHub Actions with multi-OS testing
- ✅ **Code Quality**: Ruff linting, type hints, 70% test coverage
- ✅ **Profiling**: Training bottleneck identification
- ✅ **Web Frontend**: Interactive API testing interface
- ✅ **Load Testing**: Locust for performance analysis

## Architecture

```
Development → Pre-commit Checks → GitHub Push
     ↓
CI/CD Pipelines (tests, linting, Docker builds)
     ↓
Training (Hydra configs + W&B tracking)
     ↓
API Deployment (FastAPI + ONNX optimization)
     ↓
Monitoring (Drift detection + Data collection)
```

## Key Components

| Component | Purpose |
|-----------|---------|
| `src/energy_dissagregation_mlops/` | Core ML pipeline |
| `app/main.py` | FastAPI application |
| `configs/` | Hydra experiment configurations |
| `dockerfiles/` | Container definitions (api, train, cli) |
| `frontend/` | Web UI for API testing |
| `tests/` | Unit & integration tests (70% coverage) |
| `.github/workflows/` | CI/CD pipelines |

## Documentation Sections

- [Getting Started](getting-started.md) - Installation and setup
- [CLI Usage](cli-usage.md) - Command-line interface
- [API Reference](api-reference.md) - FastAPI endpoints
- [Training](training.md) - Model training guide
- [Deployment](deployment.md) - Production deployment
- [Docker](docker.md) - Containerization
- [Drift Detection](drift-detection.md) - Monitoring strategies

## Technologies

**ML & Data**: PyTorch, ONNX, NumPy, Pandas
**API**: FastAPI, Uvicorn, Pydantic
**DevOps**: Docker, GitHub Actions, pre-commit
**Monitoring**: Weights & Biases, Loguru, Locust
**Config**: Hydra

## Testing

```bash
# Run tests
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=html

# Lint code
ruff check .

# Format code
ruff format .
```

## Load Testing

```bash
locust -f loadtest/locustfile.py --host http://localhost:8000
```

## Project Stats

- **10+ test files** with 70% code coverage
- **6 GitHub Actions workflows** for CI/CD
- **4 Docker images** (api, train, dev, cli)
- **9 Hydra configs** for experiment management
- **Drift detection** with 3 statistical tests
- **ONNX optimization** reduces inference by 30-40%

## System Requirements

- Python 3.12+
- CUDA 11.8+ (optional for GPU)
- Docker 20.10+ (optional)
- 4GB RAM minimum, 8GB+ recommended

## Troubleshooting

### Missing Data
```bash
# Download and preprocess UK-DALE dataset
python -m energy_dissagregation_mlops.cli preprocess --data-path data/raw/ukdale.h5
```

### GPU Issues
```bash
# Use CPU instead
python scripts/run_experiment.py --config-name quick_test device=cpu
```

### Docker Build Fails
```bash
# Clear cache and rebuild
docker builder prune
docker build -f dockerfiles/api.dockerfile --no-cache -t energy-api .
```

## Contributing

1. Create feature branch: `git checkout -b feature/name`
2. Follow code style (enforced by pre-commit)
3. Write tests for new code
4. Create PR against main

## License

MIT License - see LICENSE file

---

**Status**: ✅ Production Ready | **Coverage**: 70% | **Last Updated**: January 2026
