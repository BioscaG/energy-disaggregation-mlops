# Docker Quick Reference

## File Structure

```
.
├── Dockerfile              # Production CLI container
├── Dockerfile.dev          # Development container with test tools
├── docker-compose.yml      # Multi-service orchestration
├── .dockerignore           # Files excluded from Docker builds
├── .docker-env             # Environment variable template
└── dockerfiles/
    ├── api.dockerfile      # FastAPI service
    ├── train.dockerfile    # Training service
    └── cli.dockerfile      # CLI interface
```

## Quick Start

### 1. Build all images
```bash
docker-compose build
```

### 2. Start API service
```bash
docker-compose up api
```

### 3. Run training
```bash
docker-compose --profile training up train
```

### 4. Use CLI tools
```bash
docker-compose --profile cli run cli download --target-dir data/raw
docker-compose --profile cli run cli preprocess --data-path data/raw/ukdale.h5 --output-folder data/processed
docker-compose --profile cli run cli train --preprocessed-folder data/processed
```

## Services Overview

| Service | Purpose | Image | Port |
|---------|---------|-------|------|
| **api** | FastAPI inference server | energy-disaggregation-api | 8000 |
| **train** | Model training pipeline | energy-disaggregation-train | - |
| **cli** | Command-line tools | energy-disaggregation-cli | - |

## Common Commands

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs -f api
docker-compose logs -f train

# Stop containers
docker-compose stop

# Remove containers and networks
docker-compose down

# Development shell
docker-compose --profile cli run cli bash
```

## Features

✅ Multi-stage containerization for different workflows
✅ Volume mounts for persistent data
✅ Health checks on API service
✅ Development container with testing tools
✅ Docker Compose for easy orchestration
✅ .dockerignore for optimized builds
✅ Unbuffered Python output for real-time logs

## Documentation

See [docs/DOCKER.md](docs/DOCKER.md) for detailed documentation.
