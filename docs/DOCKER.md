# Docker Setup and Usage Guide

This document describes the Docker configurations for the energy disaggregation MLOps project.

## Overview

The project includes multiple Docker configurations to support different workflows:

- **api.dockerfile** - FastAPI service for inference and model serving
- **train.dockerfile** - Training service for model training
- **cli.dockerfile** - CLI interface for data processing and management
- **Dockerfile** - Production-ready CLI container
- **Dockerfile.dev** - Development container with testing tools

## Building Images

### Individual Dockerfiles

Build specific services:

```bash
# API service
docker build -f dockerfiles/api.dockerfile -t energy-disaggregation-api .

# Training service
docker build -f dockerfiles/train.dockerfile -t energy-disaggregation-train .

# CLI service
docker build -f dockerfiles/cli.dockerfile -t energy-disaggregation-cli .

# Development container
docker build -f Dockerfile.dev -t energy-disaggregation-dev .
```

### Using Docker Compose

Build all services:

```bash
docker-compose build
```

Build specific service:

```bash
docker-compose build api
```

## Running Containers

### API Service

Start the API server:

```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/configs:/app/configs:ro \
  energy-disaggregation-api
```

Or with Docker Compose:

```bash
docker-compose up api
```

The API will be available at `http://localhost:8000`.

### Training Service

Run training:

```bash
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/configs:/app/configs:ro \
  energy-disaggregation-train
```

Or with Docker Compose:

```bash
docker-compose --profile training up train
```

To run training in the background:

```bash
docker-compose --profile training up -d train
```

### CLI Service

Execute CLI commands:

```bash
docker run -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  energy-disaggregation-cli download --target-dir data/raw
```

Or with Docker Compose:

```bash
docker-compose --profile cli run cli download --target-dir data/raw
docker-compose --profile cli run cli preprocess --data-path data/raw/ukdale.h5 --output-folder data/processed
docker-compose --profile cli run cli train --preprocessed-folder data/processed
```

### Development Container

Start an interactive development shell:

```bash
docker run -it \
  -v $(pwd):/app \
  -v $(pwd)/data:/app/data \
  energy-disaggregation-dev
```

Then inside the container:

```bash
# Run tests
pytest tests/

# Format code
ruff format .

# Lint code
ruff check . --fix

# Run pre-commit hooks
pre-commit run --all-files
```

## Docker Compose Profiles

The docker-compose.yml uses profiles to organize services:

- **api** (default) - API service always starts
- **training** - Training service (opt-in with `--profile training`)
- **cli** - CLI service (opt-in with `--profile cli`)

### Examples

Start only API:

```bash
docker-compose up api
```

Start API and training:

```bash
docker-compose --profile training up
```

Start all services:

```bash
docker-compose --profile training --profile cli up
```

## Volume Mounts

All services use the same volume structure:

| Container Path | Host Path | Purpose | Mode |
|---|---|---|---|
| `/app/data` | `./data` | Dataset storage | rw |
| `/app/models` | `./models` | Model files | rw (train), ro (api) |
| `/app/configs` | `./configs` | Configuration files | ro |

- **rw** = read-write (data flows in and out)
- **ro** = read-only (data flows in, or kept stateless)

## Network

All services connect via the `energy-network` bridge network, allowing communication between containers using service names as hostnames.

## Environment Variables

Both docker-compose and manual runs set:

- `PYTHONUNBUFFERED=1` - Unbuffered Python output for real-time logs

## Best Practices

1. **Always use volumes** for data and models to persist between runs
2. **Use docker-compose** for multi-service orchestration
3. **Keep API readonly** for models to prevent accidental overwrites
4. **Run training separately** to avoid resource conflicts
5. **Development container** includes all dev dependencies

## Troubleshooting

### API not responding

Check health status:

```bash
curl http://localhost:8000/health
```

View logs:

```bash
docker-compose logs api
```

### Training container exits immediately

Check logs for errors:

```bash
docker-compose --profile training logs train
```

Ensure data volumes exist:

```bash
ls -la data/processed/
```

### Permission denied errors

Ensure proper volume permissions:

```bash
chmod -R 755 data/
chmod -R 755 models/
```

### Out of memory

Reduce batch size or use resource limits:

```yaml
services:
  train:
    # ...
    deploy:
      resources:
        limits:
          memory: 4G
```

## Production Considerations

For production deployment:

1. Use `.dockerignore` to exclude unnecessary files (already included)
2. Run API with proper logging and monitoring
3. Use environment-specific configs
4. Implement health checks (already in api.dockerfile)
5. Use secrets management for sensitive data
6. Set proper resource limits in docker-compose

## References

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Python Docker Best Practices](https://docs.docker.com/language/python/build-images/)
