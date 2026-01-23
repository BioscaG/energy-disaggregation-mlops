# Deployment Guide

## Overview

This guide covers deploying the energy disaggregation model to production environments, including local serving, containerized deployment, and cloud options.

## Local Deployment

### Quick Start (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access API at: `http://localhost:8000`

**Swagger docs**: `http://localhost:8000/docs`

### Production Server (Gunicorn)

```bash
# Install Gunicorn
pip install gunicorn

# Start with multiple workers
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

**Configuration for 16-core server**:
```bash
gunicorn app.main:app \
  --workers 8 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --worker-connections 1000 \
  --max-requests 10000 \
  --max-requests-jitter 1000
```

### Docker Deployment

#### Build Image

```bash
# Build API container
docker build -t energy-disaggregation-api:latest \
  -f dockerfiles/api.dockerfile .

# Build with specific version
docker build -t energy-disaggregation-api:v1.0 \
  -f dockerfiles/api.dockerfile .

# Push to registry
docker tag energy-disaggregation-api:latest ghcr.io/username/energy-disaggregation-api:latest
docker push ghcr.io/username/energy-disaggregation-api:latest
```

#### Run Container

```bash
# Basic
docker run -p 8000:8000 energy-disaggregation-api:latest

# With volume mount
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  energy-disaggregation-api:latest

# With environment variables
docker run -p 8000:8000 \
  -e MODEL_PATH=/app/models/best.pt \
  -e USE_ONNX=true \
  energy-disaggregation-api:latest

# Production (detached)
docker run -d \
  --name energy-api \
  --restart always \
  -p 8000:8000 \
  -v /data/models:/app/models \
  energy-disaggregation-api:latest
```

#### Docker Compose

```yaml
# docker-compose.yml
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
    environment:
      - MODEL_PATH=/app/models/best.pt
      - USE_ONNX=true
      - LOG_LEVEL=INFO
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
```

**Start deployment**:
```bash
docker-compose up -d
```

## Cloud Deployment

### Google Cloud Platform (GCP)

#### Cloud Run

Fastest deployment for containerized apps:

```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/energy-disaggregation-api

# Deploy to Cloud Run
gcloud run deploy energy-disaggregation-api \
  --image gcr.io/PROJECT_ID/energy-disaggregation-api \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated
```

**Endpoint**: `https://energy-disaggregation-api-*.run.app`

#### Compute Engine

For persistent instance with GPU:

```bash
# Create VM instance with GPU
gcloud compute instances create energy-api \
  --image-family debian-11 \
  --image-project debian-cloud \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-k80,count=1 \
  --zone us-central1-a

# SSH and deploy
gcloud compute ssh energy-api --zone us-central1-a

# On instance
git clone https://github.com/username/energy-disaggregation-mlops
cd energy-disaggregation-mlops
pip install -r requirements.txt
gunicorn app.main:app --bind 0.0.0.0:8000
```

#### Vertex AI

For managed ML service:

```bash
# Train model on Vertex AI
gcloud ai training-jobs create --region us-central1 \
  --display-name energy-disaggregation-training \
  --worker-pool-spec machine-type=n1-standard-8,accelerator-type=nvidia-tesla-k80

# Deploy as endpoint
gcloud ai endpoints deploy-model energy-disaggregation-endpoint \
  --model=energy-disaggregation-model
```

### AWS Deployment

#### Elastic Container Service (ECS)

```bash
# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login \
  --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker tag energy-disaggregation-api:latest \
  ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/energy-disaggregation-api:latest

docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/energy-disaggregation-api:latest

# Create ECS task
aws ecs register-task-definition \
  --cli-input-json file://task-definition.json

# Start ECS service
aws ecs create-service \
  --cluster energy-cluster \
  --service-name energy-api \
  --task-definition energy-disaggregation-api:1
```

#### SageMaker

```bash
# Build and push container
aws sagemaker create-model \
  --model-name energy-disaggregation \
  --containers Image=ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/energy-disaggregation-api:latest

# Deploy endpoint
aws sagemaker create-endpoint \
  --endpoint-name energy-disaggregation-endpoint \
  --endpoint-config-name energy-disaggregation-config
```

## Reverse Proxy (Nginx)

### Basic Configuration

```nginx
# nginx.conf
upstream api {
    server api:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.example.com;

    # Increase timeouts for inference
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;

    location / {
        proxy_pass http://api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Health check
        access_log /var/log/nginx/api_access.log;
        error_log /var/log/nginx/api_error.log;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://api;
        access_log off;
    }
}
```

### HTTPS with SSL

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /etc/ssl/certs/server.crt;
    ssl_certificate_key /etc/ssl/private/server.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://api;
        proxy_set_header X-Forwarded-Proto https;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.example.com;
    return 301 https://$server_name$request_uri;
}
```

## Load Balancing

### Multiple API Instances

```nginx
upstream api_cluster {
    least_conn;  # Use least connections algorithm

    server api1:8000 weight=1;
    server api2:8000 weight=1;
    server api3:8000 weight=1;

    # Health checks
    check interval=3000 rise=2 fall=5 timeout=1000;
}

server {
    listen 80;
    location / {
        proxy_pass http://api_cluster;
    }
}
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
  energy-disaggregation-api:latest

# Scale service
docker service scale energy-api=5
```

## Monitoring & Logging

### Health Checks

```bash
# Continuous monitoring
while true; do
  curl -f http://localhost:8000/health || echo "API is down!"
  sleep 10
done
```

### Log Aggregation

```yaml
# With ELK Stack
filebeat:
  inputs:
    - type: log
      enabled: true
      paths:
        - /var/log/api/*.log

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

### Metrics Collection

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, start_http_server

request_count = Counter('api_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('api_request_duration_seconds', 'Request duration')

@app.post('/predict')
def predict(x):
    with request_duration.time():
        result = model(x)
        request_count.labels(method='POST', endpoint='/predict').inc()
        return result
```

## Database Persistence

### Store Predictions

```python
from sqlalchemy import create_engine, Column, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    input_power = Column(Float)
    predicted_power = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Save predictions
engine = create_engine('postgresql://user:pass@localhost/energy_db')
Base.metadata.create_all(engine)

session = Session(engine)
session.add(Prediction(input_power=0.5, predicted_power=0.42))
session.commit()
```

## Performance Optimization

### Model Optimization

1. **Use ONNX**:
   ```python
   # 30-40% faster inference
   import onnxruntime
   session = onnxruntime.InferenceSession("models/model.onnx")
   ```

2. **Quantization**:
   ```python
   # 4x faster, minimal accuracy loss
   from torch.quantization import quantize_dynamic
   quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
   ```

3. **Batching**:
   ```python
   # Process 32 samples instead of 1
   # 20-30% throughput increase
   predictions = model(batch_x)
   ```

### Infrastructure Optimization

1. **Use GPU** for inference (3-5x speedup)
2. **Redis caching** for frequent inputs
3. **CDN** for static files
4. **Database connection pooling**
5. **Request queuing** for burst traffic

## Security

### Authentication

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post('/predict')
def predict(x: List[float], credentials: HTTPAuthCredentials = Depends(security)):
    # Verify token
    if not verify_token(credentials.credentials):
        raise HTTPException(status_code=403, detail="Invalid token")
    return model(x)
```

### Rate Limiting

```bash
# nginx rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

server {
    location /predict {
        limit_req zone=api_limit burst=20;
        proxy_pass http://api;
    }
}
```

### Input Validation

Already implemented in `app.main.py` with Pydantic models.

---

**Next Steps**:
- Setup [Drift Detection](drift-detection.md) for production monitoring
- Create [CI/CD Pipeline](../DOCKER.md) for automated deployments
- Review [API Reference](api-reference.md) for endpoint details
