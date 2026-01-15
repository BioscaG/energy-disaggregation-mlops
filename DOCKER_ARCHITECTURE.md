# Docker Architecture Overview

## Project Structure

```
energy-disaggregation-mlops/
├── Dockerfile                    # Production CLI
├── Dockerfile.dev                # Development environment
├── docker-compose.yml            # Multi-service orchestration
├── .dockerignore                 # Exclude from builds
├── .docker-env                   # Environment configuration template
├── Makefile.docker               # Convenient make commands
│
├── dockerfiles/
│   ├── api.dockerfile            # FastAPI service (port 8000)
│   ├── train.dockerfile          # Training pipeline
│   └── cli.dockerfile            # CLI interface
│
├── docs/
│   └── DOCKER.md                 # Comprehensive Docker guide
│
├── DOCKER_IMPLEMENTATION.md      # Implementation details
├── DOCKER_QUICKSTART.md          # Quick reference guide
│
├── src/
│   └── energy_dissagregation_mlops/
│       ├── api.py                # API endpoints
│       ├── train.py              # Training logic
│       ├── cli.py                # CLI commands
│       └── ...
│
├── data/
│   ├── raw/                      # Raw dataset
│   └── processed/                # Preprocessed data
│
└── models/                       # Trained model files
```

## Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Docker Compose Network                    │
│                   (energy-network bridge)                   │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
    │  API Service│   │   Training  │   │  CLI Service │
    │  (Port 8000)│   │  (Background)   │ (Interactive)│
    ├─────────────┤   ├─────────────┤   ├──────────────┤
    │ api.docker  │   │train.docker │   │cli.dockerfile│
    │   (22 MB)   │   │   (22 MB)   │   │   (22 MB)    │
    └─────────────┘   └─────────────┘   └──────────────┘
         │                    │                    │
    ┌────┴─────────────┬─────┴────────────┬───────┴─────┐
    │                  │                  │             │
    ▼                  ▼                  ▼             ▼
  /models        /data & /models    /data & /models  /data
  (read-only)   (read-write)      (read-write)    (interactive)
```

## Container Images

| Image | Size | Purpose | Features |
|-------|------|---------|----------|
| **energy-disaggregation-api** | ~350MB | FastAPI server | Health checks, port 8000 |
| **energy-disaggregation-train** | ~350MB | Model training | GPU support ready |
| **energy-disaggregation-cli** | ~350MB | Data processing | Interactive CLI |
| **energy-disaggregation-dev** | ~420MB | Development | Testing & debugging tools |

## Workflow Diagrams

### Training Workflow
```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
│   Download  │────→│  Preprocess  │────→│    Train     │────→│  Models  │
│   Dataset   │     │    Data      │     │   Pipeline   │     │  Output  │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────┘
   CLI Service         CLI Service        Training Service       Volumes
```

### Inference Workflow
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Load Model  │────→│   Process    │────→│   Return     │
│   from Path  │     │   Request    │     │   Prediction │
└──────────────┘     └──────────────┘     └──────────────┘
   API Service         API Service         API Service
   (Port 8000)         (Port 8000)         (Port 8000)
```

## Data Flow

```
Host Machine                          Containers
┌──────────────┐                    ┌──────────────┐
│ data/raw/    │◄─Volume Mount──────┤ /app/data    │
│ ukdale.h5    │                    │              │
└──────────────┘                    └──────────────┘
                                     Training, CLI
       │
       │ Download
       ▼
┌──────────────┐                    ┌──────────────┐
│ data/        │◄─Volume Mount──────┤ /app/data    │
│ processed/   │                    │              │
└──────────────┘                    └──────────────┘
                                     All services
       │
       │ Train
       ▼
┌──────────────┐                    ┌──────────────┐
│ models/      │◄─Volume Mount──────┤ /app/models  │
│ best.pt      │ (Read-write API)   │              │
└──────────────┘ (Read-only Train)  └──────────────┘
                                     All services
```

## Docker Compose Profiles

```
Default Profile (always runs):
├── api          Ready for inference requests

With --profile training:
├── api          Ready for inference requests
└── train        Long-running training job

With --profile cli:
├── api          Ready for inference requests
└── cli          Interactive shell for commands

With --profile training --profile cli:
├── api          Ready for inference requests
├── train        Long-running training job
└── cli          Interactive shell for commands
```

## Quick Command Reference

```bash
# Build
make -f Makefile.docker build

# Start services
make -f Makefile.docker up-api          # API only
make -f Makefile.docker up-train        # Training
make -f Makefile.docker up              # All services

# Common workflows
make -f Makefile.docker download-data   # Download dataset
make -f Makefile.docker preprocess      # Preprocess
make -f Makefile.docker train           # Train model
make -f Makefile.docker serve           # Start API
make -f Makefile.docker dev             # Dev shell

# Debugging
make -f Makefile.docker logs-api        # View logs
make -f Makefile.docker test            # Run tests
```

## Environment Variables

```env
# Python
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Training
TRAIN_EPOCHS=5
TRAIN_BATCH_SIZE=32
TRAIN_LR=0.001

# Data Paths
DATA_PATH=/app/data
MODELS_PATH=/app/models
CONFIGS_PATH=/app/configs
```

## Key Features

✅ **Multi-Container Architecture** - Separate concerns
✅ **Health Monitoring** - API health checks
✅ **Persistent Volumes** - Data & model persistence
✅ **Network Isolation** - Dedicated bridge network
✅ **Development Support** - Full dev container
✅ **Easy Orchestration** - Docker Compose
✅ **Production Ready** - Optimized images, health checks
✅ **Comprehensive Documentation** - Multiple guides

## Next Steps

1. **Build images**: `make -f Makefile.docker build`
2. **Start API**: `make -f Makefile.docker up-api`
3. **Download data**: `make -f Makefile.docker download-data`
4. **Train model**: `make -f Makefile.docker train`
5. **Serve predictions**: `make -f Makefile.docker serve`

See [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) for quick reference.
See [docs/DOCKER.md](docs/DOCKER.md) for comprehensive guide.
