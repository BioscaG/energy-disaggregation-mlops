# API Reference

## Overview

The FastAPI application provides REST endpoints for model inference, health checking, and monitoring.

**Base URL**: `http://localhost:8000`

## Endpoints

### Health Check

Check if the API is running and model is loaded.

```http
GET /health
```

**Response** (200 OK):
```json
{
  "status": "ok",
  "model_loaded": true,
  "timestamp": "2026-01-23T12:00:00Z"
}
```

**Use case**: Load balancer health checks, service monitoring

---

### Predict (PyTorch)

Make predictions using the PyTorch model.

```http
POST /predict
Content-Type: application/json

{
  "x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}
```

**Request Parameters**:

- `x` (array, required): Time-series input of shape `[T]` or `[B, T]`
  - Single sample: `[0.1, 0.2, ..., 1.0]` (shape [T])
  - Batch: `[[0.1, 0.2, ...], [0.3, 0.4, ...]]` (shape [B, T])
  - Range: Normalized power consumption (0-1 typically)

**Response** (200 OK):
```json
{
  "batch_size": 1,
  "t": 10,
  "y": [0.245, 0.318, 0.412, 0.521, 0.643, 0.771, 0.901, 0.023, 0.145, 0.267]
}
```

**Response Fields**:

- `batch_size`: Number of samples in batch
- `t`: Sequence length (timesteps)
- `y`: Predicted appliance power consumption

**Error Responses**:

- 400 Bad Request: Empty input or invalid format
- 500 Internal Server Error: Model prediction failed

**Latency**: ~50-100ms per sample on CPU

**Use case**: Appliance power disaggregation, energy monitoring

---

### Predict (ONNX)

Make predictions using the ONNX-optimized model (30-40% faster).

```http
POST /predict/onnx
Content-Type: application/json

{
  "x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}
```

**Request**: Same as `/predict`

**Response**: Same as `/predict`

**Latency**: ~15-30ms per sample on CPU (3-5x faster)

**Use case**: High-throughput inference, edge deployment

---

## Examples

### Python Client

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Single prediction
data = {"x": [0.1, 0.2, 0.3, 0.4, 0.5]}
response = requests.post(f"{BASE_URL}/predict", json=data)
predictions = response.json()
print(f"Predicted consumption: {predictions['y']}")

# Batch prediction
data = {
    "x": [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7],
    ]
}
response = requests.post(f"{BASE_URL}/predict", json=data)
predictions = response.json()
print(f"Batch size: {predictions['batch_size']}")
print(f"Predictions: {predictions['y']}")
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"x": [0.1, 0.2, 0.3, 0.4, 0.5]}'

# Batch prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "x": [
      [0.1, 0.2, 0.3],
      [0.2, 0.3, 0.4]
    ]
  }'

# ONNX prediction
curl -X POST http://localhost:8000/predict/onnx \
  -H "Content-Type: application/json" \
  -d '{"x": [0.1, 0.2, 0.3, 0.4, 0.5]}'
```

### JavaScript/Fetch

```javascript
// Health check
fetch('http://localhost:8000/health')
  .then(r => r.json())
  .then(data => console.log(data));

// Make prediction
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({x: [0.1, 0.2, 0.3, 0.4, 0.5]})
})
  .then(r => r.json())
  .then(data => console.log(`Predictions: ${data.y}`));
```

## Performance

### Throughput

| Model | Backend | Latency (ms) | Throughput (req/s) |
|-------|---------|-------------|-------------------|
| Single | PyTorch | 50-100 | 10-20 |
| Single | ONNX | 15-30 | 33-67 |
| Batch (32) | PyTorch | 1500-2000 | 16-21 |
| Batch (32) | ONNX | 400-600 | 53-80 |

### Resource Usage

| Component | Memory | CPU |
|-----------|--------|-----|
| API (idle) | 200MB | 5% |
| API + PyTorch Model | 500MB | 15% |
| API + ONNX Model | 300MB | 15% |
| Prediction (batch 32) | +50MB | 80-100% |

## Data Format

### Input Format

**Single sample** (shape [T]):
```json
{"x": [0.1, 0.2, 0.3, 0.4, 0.5]}
```

**Batch** (shape [B, T]):
```json
{
  "x": [
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.6, 0.7, 0.8, 0.9, 1.0]
  ]
}
```

### Output Format

```json
{
  "batch_size": 2,
  "t": 5,
  "y": [
    [0.245, 0.318, 0.412, 0.521, 0.643],
    [0.771, 0.901, 0.023, 0.145, 0.267]
  ]
}
```

## Error Handling

### 400 Bad Request

```json
{
  "detail": "Empty input rejected"
}
```

**Causes**:
- Empty input array
- Invalid JSON format
- Missing required fields

### 503 Service Unavailable

```json
{
  "detail": "Model not loaded"
}
```

**Causes**:
- Model file missing
- CUDA out of memory
- Server startup failed

## Rate Limiting

No built-in rate limiting (can be added via middleware).

**Recommended**:
- Nginx reverse proxy with rate limits
- Application-level token bucket
- Deploy multiple API instances

## CORS

Not enabled by default. To enable:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Monitoring

### Logging

All requests logged via Loguru:
```
2026-01-23 12:00:00 | INFO | POST /predict - batch_size=1, latency=45ms
2026-01-23 12:00:01 | INFO | POST /predict/onnx - batch_size=1, latency=12ms
```

### Metrics

Tracked metrics:
- Request count per endpoint
- Response latency (min/max/mean)
- Error rates
- Model latency breakdown

Access via monitoring dashboard (if configured).

## Versioning

Current API version: **1.0**

No breaking changes planned. Future versions will maintain backward compatibility.

---

**Note**: Default model uses PyTorch. ONNX model available after export via `scripts/export_onnx.py`.
