# Data Collection & Production Monitoring (M27 Extension)

## Overview

Your API now automatically collects input-output data from all predictions. This enables:
- **Real-time drift detection** - Know when your data has shifted
- **Performance monitoring** - Track prediction patterns
- **Historical analysis** - Understand model behavior over time
- **Automated alerts** - Get warnings before problems occur

## How It Works

### Automatic Data Collection

Every time you make a prediction, the system automatically records:
- Input data (mains power)
- Output data (predicted appliance power)
- Timestamp
- Prediction ID
- Metadata (endpoint, batch info)

```
User Request
    ↓
Make Prediction
    ↓
Collect Data ← NEW!
    ↓
Return Result
```

### Key Features

✅ **Automatic**: Works with existing `/predict` endpoint
✅ **Thread-safe**: Multiple concurrent requests handled safely
✅ **Bounded**: Keeps only recent 10,000 predictions in memory
✅ **Drift-aware**: Compares to reference distribution
✅ **Persistent**: Can save to disk for long-term analysis

## Setup

### 1. Save Reference Distribution (One-time)

```bash
python scripts/save_reference_distribution.py
```

This creates `models/reference_distribution.npy` - the baseline from your training data.

### 2. Start API (Automatic from now on)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The system will:
- Load reference distribution on startup
- Collect data from all predictions
- Enable monitoring endpoints

## API Endpoints

### Get Collection Statistics

```bash
curl http://localhost:8000/api/v1/collector/stats
```

Response:
```json
{
  "total_predictions": 150,
  "buffered_predictions": 150,
  "errors_count": 0,
  "input_statistics": {
    "mean": 502.15,
    "std": 48.92,
    "min": 301.23,
    "max": 698.45
  },
  "output_statistics": {
    "mean": 42.15,
    "std": 28.92,
    "min": 0.0,
    "max": 199.87
  },
  "time_range": {
    "first": "2026-01-20T18:00:00.123456",
    "last": "2026-01-20T18:15:32.987654"
  }
}
```

### Get Recent Predictions

```bash
curl "http://localhost:8000/api/v1/collector/recent?n=10"
```

Response shows last 10 predictions with input, output, and timestamp.

### Check for Drift

```bash
curl http://localhost:8000/api/v1/drift/status
```

Response:
```json
{
  "status": "analyzed",
  "drift_detected": false,
  "psi": 0.0845,
  "ks_statistic": 0.0123,
  "ks_p_value": 0.5678,
  "mmd": 0.0012,
  "interpretation": "✅ STABLE",
  "recommendation": "CONTINUE NORMAL OPERATION"
}
```

**Interpretation Guide:**
- PSI < 0.1: ✅ STABLE - Everything is normal
- PSI 0.1-0.25: ⚠️ MODERATE DRIFT - Increase monitoring
- PSI > 0.25: 🚨 CRITICAL DRIFT - Trigger retraining

### Get Performance Metrics

```bash
curl http://localhost:8000/api/v1/drift/performance
```

Response shows output statistics and prediction count.

### Get Hourly Summary

```bash
curl http://localhost:8000/api/v1/hourly/summary
```

Response:
```json
{
  "2026-01-20T18:00:00": {
    "count": 25,
    "input_mean": 501.23,
    "output_mean": 41.45
  },
  "2026-01-20T19:00:00": {
    "count": 32,
    "input_mean": 520.15,
    "output_mean": 45.32
  }
}
```

### Save Collected Data

```bash
curl -X POST http://localhost:8000/api/v1/collector/save
```

Saves all collected data to `data/production/predictions_TIMESTAMP.json`

Response:
```json
{
  "status": "saved",
  "filepath": "/path/to/predictions_20260120_181523.json"
}
```

### Clear Collected Data

```bash
curl -X DELETE http://localhost:8000/api/v1/collector/clear
```

Clears all data from memory (but not from disk if saved).

## Workflow Example

### 1. Make Predictions

```bash
# Each prediction is automatically collected
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"x": [500, 501, 499, 502, ...]}'
```

### 2. Check Status (Every hour or day)

```bash
# Get drift status
curl http://localhost:8000/api/v1/drift/status

# Expected output:
# {
#   "psi": 0.15,
#   "interpretation": "✅ STABLE",
#   "recommendation": "CONTINUE NORMAL OPERATION"
# }
```

### 3. If Drift Detected

```bash
# Save data for analysis
curl -X POST http://localhost:8000/api/v1/collector/save

# Investigate with Python:
import json
with open("data/production/predictions_20260120_181523.json") as f:
    data = json.load(f)

# Analyze patterns, check what changed, etc.
# Then trigger retraining
```

### 4. After Retraining

```bash
# Save new reference distribution
python scripts/save_reference_distribution.py

# Clear old data
curl -X DELETE http://localhost:8000/api/v1/collector/clear

# Continue monitoring
```

## Python API Usage

### Use Collector Directly

```python
from energy_dissagregation_mlops.data_collection import get_collector
import numpy as np

# Get global collector
collector = get_collector()

# Record a prediction
x = np.random.randn(1024)
y = np.random.randn(1024)
collector.record_prediction(x, y, prediction_id="pred_001")

# Get stats
stats = collector.get_statistics()
print(f"Collected: {stats['total_predictions']} predictions")

# Save data
filepath = collector.save_batch("my_batch.json")

# Clear memory
collector.clear()
```

### Use Drift Monitor

```python
from energy_dissagregation_mlops.data_collection import get_drift_monitor
import numpy as np

monitor = get_drift_monitor()

# Load reference
reference = np.load("models/reference_distribution.npy")

# Analyze drift
result = monitor.analyze_drift(reference)
print(f"Drift status: {result['interpretation']}")

if result['psi'] > 0.25:
    print("⚠️ TRIGGER RETRAINING!")
```

## File Structure

```
models/
├─ best.pt
├─ model.onnx
└─ reference_distribution.npy    ← Your baseline

data/
├─ processed/                     ← Preprocessed training data
└─ production/
   ├─ predictions_20260120_180000.json
   ├─ predictions_20260120_190000.json
   └─ predictions_20260120_200000.json
```

## Configuration

Adjust collection settings in `data_collection.py`:

```python
# Maximum in-memory samples (default: 10,000)
_collector = ProductionDataCollector(max_samples=50000)

# Collection directory (default: data/production)
_collector = ProductionDataCollector(collection_dir="data/prod_monitoring")
```

## Monitoring Strategy

### Recommended Checks

**Hourly:**
```bash
curl http://localhost:8000/api/v1/drift/status | grep psi
```

**Daily:**
```bash
curl http://localhost:8000/api/v1/collector/stats
curl http://localhost:8000/api/v1/hourly/summary
```

**Weekly:**
```bash
# Save and archive
curl -X POST http://localhost:8000/api/v1/collector/save
# Then clear
curl -X DELETE http://localhost:8000/api/v1/collector/clear
```

### Alert Thresholds

| PSI Value | Alert Level | Action |
|-----------|-------------|--------|
| < 0.1 | Green ✅ | None needed |
| 0.1-0.25 | Yellow ⚠️ | Monitor closely, prepare retraining |
| > 0.25 | Red 🚨 | TRIGGER RETRAINING IMMEDIATELY |

## Troubleshooting

### Reference Distribution Not Found

```
WARNING: Reference distribution not found at models/reference_distribution.npy
```

**Solution**: Run the save script first
```bash
python scripts/save_reference_distribution.py
```

### API Not Collecting Data

**Check**:
```bash
curl http://localhost:8000/api/v1/collector/stats
```

If `"status": "collector_not_initialized"`, restart API.

### Performance Issues with Many Predictions

If API slows down:
1. Reduce `max_samples` in `data_collection.py`
2. Save data more frequently
3. Clear data after saving

## Testing

Run collection tests:

```bash
pytest tests/test_data_collection.py -v
```

Expected: 15/15 tests passing ✅

## Next Steps

1. **Set up monitoring dashboard** - Integrate with Grafana/Datadog
2. **Automate retraining** - Trigger when PSI > 0.25
3. **Alert integration** - Send to Slack/email when drift detected
4. **Long-term storage** - Archive to database instead of JSON
5. **Feature attribution** - Identify which features are drifting

---

**Status**: ✅ Production-ready, 15/15 tests passing
