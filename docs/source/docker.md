# Docker

## Overview

Docker containers provide reproducible environments for training, inference, and development. This guide covers building and running Docker images.

## Images

### API Image

FastAPI server for production inference.

```dockerfile
# dockerfiles/api.dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

# Start API
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build**:
```bash
docker build -t energy-disaggregation-api:latest -f dockerfiles/api.dockerfile .
```

**Run**:
```bash
docker run -p 8000:8000 energy-disaggregation-api:latest
```

**With GPU**:
```bash
docker run --gpus all -p 8000:8000 energy-disaggregation-api:latest
```

### Training Image

Full training environment with CUDA support.

```dockerfile
# dockerfiles/train.dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements_dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements_dev.txt

# Copy project
COPY . .

# Setup entry point
ENTRYPOINT ["python"]
CMD ["scripts/run_experiment.py"]
```

**Build**:
```bash
docker build -t energy-disaggregation-train:latest -f dockerfiles/train.dockerfile .
```

**Run training**:
```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/wandb:/app/wandb \
  energy-disaggregation-train:latest \
  scripts/run_experiment.py --config-name normal_training
```

### CLI Image

Command-line interface for running arbitrary commands.

```dockerfile
# dockerfiles/cli.dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python"]
```

**Build**:
```bash
docker build -t energy-disaggregation-cli:latest -f dockerfiles/cli.dockerfile .
```

**Run command**:
```bash
docker run -v $(pwd)/data:/app/data energy-disaggregation-cli:latest \
  scripts/download_dataset.py
```

## Docker Compose

Multi-container deployment with API, Redis cache, and database.

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: dockerfiles/api.dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/models/best.pt
      - USE_ONNX=true
      - LOG_LEVEL=INFO
    restart: always
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: always

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: energy
      POSTGRES_PASSWORD: changeme
      POSTGRES_DB: energy_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: always

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    restart: always

volumes:
  redis_data:
  postgres_data:
```

**Start services**:
```bash
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## Container Registry

### Push to GitHub Container Registry (GHCR)

```bash
# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin

# Tag image
docker tag energy-disaggregation-api:latest \
  ghcr.io/username/energy-disaggregation-api:latest

# Push
docker push ghcr.io/username/energy-disaggregation-api:latest
```

### Automated Builds

GitHub Actions automatically builds and pushes on every commit:

```yaml
# .github/workflows/docker_build.yaml
name: Build Docker Images

on:
  push:
    branches: [main]
    paths:
      - 'Dockerfile*'
      - 'dockerfiles/**'
      - 'requirements.txt'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - uses: docker/build-push-action@v4
        with:
          context: .
          file: ./dockerfiles/api.dockerfile
          push: true
          tags: ghcr.io/${{ github.repository }}/api:latest
          cache-from: type=gha
          cache-to: type=gha
```

## Best Practices

### 1. Multi-stage Builds

Reduce image size:

```dockerfile
# Build stage
FROM python:3.12 AS builder
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.12-slim
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY app/ ./app/
```

### 2. .dockerignore

Exclude unnecessary files:

```
# .dockerignore
.git
.gitignore
.github
.venv
venv
__pycache__
*.pyc
.pytest_cache
.mypy_cache
wandb/
data/raw/
*.egg-info
```

### 3. Non-root User

Security best practice:

```dockerfile
FROM python:3.12-slim

RUN useradd -m appuser
USER appuser

WORKDIR /app
```

### 4. Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"
```

### 5. Resource Limits

```bash
docker run --memory="2g" --cpus="1" energy-disaggregation-api:latest
```

## Development Workflow

### Development Container

Fast iteration with mounted source code:

```bash
docker run -it \
  -v $(pwd):/app \
  -v $(pwd)/.venv:/app/.venv \
  -p 8000:8000 \
  --entrypoint bash \
  energy-disaggregation-dev:latest

# Inside container
pip install -e .
uvicorn app.main:app --reload
```

### Remote Development

Connect VS Code to container:

```json
{
  "remoteUser": "appuser",
  "image": "energy-disaggregation-api:latest",
  "forwardPorts": [8000],
  "customizations": {
    "vscode": {
      "extensions": ["ms-python.python"]
    }
  }
}
```

## Deployment Scenarios

### Single Container

```bash
docker run -d --name energy-api \
  --restart always \
  -p 8000:8000 \
  -v /data/models:/app/models \
  ghcr.io/username/energy-disaggregation-api:latest
```

### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Create service
docker service create \
  --name energy-api \
  --replicas 3 \
  --publish 8000:8000 \
  ghcr.io/username/energy-disaggregation-api:latest

# Scale
docker service scale energy-api=5
```

### Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: energy-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: energy-api
  template:
    metadata:
      labels:
        app: energy-api
    spec:
      containers:
      - name: api
        image: ghcr.io/username/energy-disaggregation-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
```

Deploy:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl logs -f deployment/energy-api
```

## Debugging

### View Logs

```bash
# Container logs
docker logs container_id

# Real-time
docker logs -f container_id

# Last 100 lines
docker logs --tail 100 container_id
```

### Shell Access

```bash
# Interactive bash
docker exec -it container_id bash

# Single command
docker exec container_id python --version
```

### Container Inspection

```bash
# View image layers
docker history energy-disaggregation-api:latest

# Image details
docker inspect energy-disaggregation-api:latest

# Running containers
docker ps -a

# Resource usage
docker stats
```

## Performance Optimization

### 1. Layer Caching

Order Dockerfile commands by change frequency:

```dockerfile
# Rarely changes → Frequently changes
FROM python:3.12-slim
RUN apt-get update && apt-get install -y ...
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ ./app/
COPY src/ ./src/
```

### 2. BuildKit

Enable advanced caching:

```bash
DOCKER_BUILDKIT=1 docker build -t energy-api:latest .
```

### 3. Secrets Management

```bash
docker build \
  --secret=wandb_api_key \
  -t energy-api:latest .
```

In Dockerfile:
```dockerfile
RUN --mount=type=secret,id=wandb_api_key \
  wandb login $(cat /run/secrets/wandb_api_key)
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Use different port
docker run -p 8001:8000 energy-disaggregation-api:latest
```

### Out of Disk Space

```bash
# Clean up dangling images
docker image prune -a

# Remove stopped containers
docker container prune

# Check space usage
docker system df
```

### Build Failures

```bash
# Verbose output
docker build --progress=plain -t energy-api:latest .

# Keep intermediate containers
docker build --no-cache -t energy-api:latest .
```

---

**Next Steps**:
- [Deployment Guide](deployment.md) for cloud deployment
- [API Reference](api-reference.md) for endpoint details
- [Training Guide](training.md) for training in containers
