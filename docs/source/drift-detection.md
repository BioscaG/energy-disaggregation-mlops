# Drift Detection

## Overview

Drift detection monitors whether the model's input or output distribution changes in production. This guide explains the drift detection system and how to use it.

## What is Drift?

**Data Drift**: Input distribution changes
- Example: Temperature range shifts, new appliance types introduced
- Impact: Model makes poorer predictions

**Model Drift**: Relationship between input/output changes
- Example: Appliance efficiency improves, affecting power consumption patterns
- Impact: Model becomes outdated

**Concept Drift**: Underlying prediction task changes
- Example: Building renovation changes power consumption patterns
- Impact: Model needs retraining

## Statistical Tests

### Kolmogorov-Smirnov (KS) Test

Measures maximum difference between two distributions:

```python
from scipy.stats import ks_2samp

ks_statistic, p_value = ks_2samp(reference, current)

# p_value < 0.05 → Drift detected
```

**Pros**:
- No distribution assumption
- Sensitive to distribution changes
- Fast computation

**Cons**:
- Only checks maximum difference
- Less sensitive to subtle shifts

**Threshold**: p_value < 0.05

### Wasserstein Distance (PSI)

Population Stability Index measures distributional shift:

```python
from scipy.stats import wasserstein_distance

distance = wasserstein_distance(reference, current)

# Interpretation:
# < 0.1:  No significant population change
# 0.1-0.25: Small population change
# > 0.25: Large population change (drift)
```

**Pros**:
- Robust to outliers
- Interpretable magnitude
- Works well for practical use

**Cons**:
- Requires binning for discrete data
- Computationally intensive for large datasets

**Threshold**: distance > 0.25

### Maximum Mean Discrepancy (MMD)

Tests if two distributions are equal using kernel methods:

```python
# Implemented in drift_detection.py
mmd_statistic = mmd_test(reference, current, kernel='rbf')

# mmd_statistic > threshold → Drift detected
```

**Pros**:
- Detects all types of differences
- Handles high-dimensional data well
- Strong theoretical foundation

**Cons**:
- Computationally expensive
- Kernel selection affects results
- Threshold selection is critical

**Threshold**: Auto-calibrated on reference data

## Using the Drift Detector

### Initialization

```python
from src.energy_dissagregation_mlops.drift_detection import DataDriftDetector
import numpy as np

# Load reference distribution (from training)
reference_data = np.load('models/reference_distribution.npy')

# Create detector
detector = DataDriftDetector(
    reference_data=reference_data,
    test_types=['ks', 'psi', 'mmd'],  # Statistical tests to use
    p_value_threshold=0.05,  # Significance level
    save_dir='drift_reports/'
)
```

### Run Detection

```python
# Current production data (recent 24 hours)
current_data = np.array([...])  # Shape: (n_samples, n_features)

# Run drift detection
drift_detected, report = detector.detect_drift(current_data)

if drift_detected:
    print(f"⚠️ DRIFT DETECTED!")
    print(report)
    # Trigger retraining
    trigger_retraining()
else:
    print(f"✓ No drift detected")
```

### Example Output

```json
{
  "timestamp": "2026-01-23T12:00:00Z",
  "drift_detected": true,
  "tests": {
    "ks_test": {
      "statistic": 0.45,
      "p_value": 0.001,
      "drifted": true
    },
    "psi_test": {
      "psi": 0.38,
      "threshold": 0.25,
      "drifted": true
    },
    "mmd_test": {
      "statistic": 1.23,
      "threshold": 0.95,
      "drifted": true
    }
  },
  "consensus": true,
  "severity": "high"
}
```

## Production Monitoring

### Real-time Monitoring

```python
from src.energy_dissagregation_mlops.data_collection import ProductionDataCollector

# Collect recent data
collector = ProductionDataCollector(db_connection_string)
recent_data = collector.get_recent_hours(hours=24)

# Run drift detection
detector = DataDriftDetector(reference_data)
drift_detected, report = detector.detect_drift(recent_data)

# Log results
if drift_detected:
    logger.warning(f"Drift detected: {report}")
    alert_team()
```

### Scheduled Checks (Daily)

```python
# monitoring_job.py
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

@scheduler.scheduled_job('cron', hour=0)  # Daily at midnight
def daily_drift_check():
    recent_data = collector.get_recent_hours(hours=24)
    drift_detected, report = detector.detect_drift(recent_data)

    if drift_detected:
        save_drift_report(report)
        notify_monitoring_dashboard()

scheduler.start()
```

### Dashboard Integration

Visualize drift over time:

```python
import plotly.graph_objects as go

# KS statistic trends
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dates,
    y=ks_statistics,
    mode='lines',
    name='KS Statistic',
    line=dict(color='blue')
))
fig.add_hline(y=0.05, line_dash="dash", line_color="red",
              annotation_text="Threshold")
fig.show()
```

## Retraining Strategy

### When to Retrain?

Retrain when:
1. **Drift detected** by all three tests (high confidence)
2. **Performance metrics degrade**: Val loss increases by >10%
3. **Business metrics suffer**: Prediction MAPE > 20%
4. **Scheduled retraining**: Every 30 days (preventive)

### Incremental Retraining

```python
# Use current production data + old training data
old_data = load_training_data()
new_data = collector.get_data_since(last_training_date)

combined_data = np.concatenate([old_data, new_data])

# Retrain with combined data
train(
    data=combined_data,
    config='incremental_training.yaml',
    wandb_tags=['drift_triggered', f'drift_severity_{severity}']
)
```

### Full Retraining

```python
# Retrain from scratch with all available data
train(
    data=all_production_data,
    config='full_training.yaml',
    reset_model=True
)
```

## Configuration

### Drift Detection Config

```yaml
# configs/drift_detection.yaml
reference_data_path: models/reference_distribution.npy

detector:
  test_types:
    - ks       # Fast, good for quick checks
    - psi      # Robust, recommended
    - mmd      # Comprehensive, slower

  thresholds:
    ks_p_value: 0.05          # P-value threshold
    psi_distance: 0.25        # PSI threshold
    mmd_statistic: auto       # Auto-calibrate on reference

  consensus_required: true     # All tests must agree
  severity_calculation: weighted

monitoring:
  enabled: true
  check_interval_hours: 6
  report_dir: drift_reports/
  alert_email: ops@example.com
```

### Custom Thresholds

```python
detector = DataDriftDetector(
    reference_data=reference_data,
    ks_p_value_threshold=0.01,      # Stricter
    psi_distance_threshold=0.15,    # More sensitive
    mmd_threshold=0.8,
    consensus_required=True
)
```

## Handling Drift

### Alert Levels

| Severity | Condition | Action |
|----------|-----------|--------|
| Low | 1-2 tests detect drift | Log and monitor |
| Medium | All tests detect mild drift | Schedule retraining |
| High | All tests detect strong drift | Retrain immediately |
| Critical | Performance drops >30% | Rollback to previous model |

### Response Procedures

```python
def handle_drift(report):
    severity = report['severity']

    if severity == 'critical':
        # Rollback immediately
        rollback_to_previous_model()
        alert_team(urgency='critical')

    elif severity == 'high':
        # Trigger retraining, use previous model temporarily
        trigger_retraining_job()
        log_incident(report)

    elif severity == 'medium':
        # Schedule retraining for next maintenance window
        schedule_retraining(urgency='high')
        log_warning(report)

    else:  # low
        # Monitor closely
        increase_monitoring_frequency()
        log_info(report)
```

## Advanced Topics

### Multivariate Drift Detection

Detect drift in multiple features simultaneously:

```python
# Input: multi-dimensional features
current_data_features = current_data[:, :n_features]
reference_features = reference_data[:, :n_features]

# Detect drift per feature
feature_names = ['appliance_power', 'temperature', 'humidity']
for i, feature_name in enumerate(feature_names):
    current_feature = current_data_features[:, i]
    ref_feature = reference_features[:, i]

    ks_stat, p_value = ks_2samp(ref_feature, current_feature)
    print(f"{feature_name}: p_value={p_value:.4f}")
```

### Drift Severity Scoring

Quantify how severe the drift is:

```python
def calculate_severity(report):
    """Severity from 0 (no drift) to 1 (severe)"""
    scores = []

    # KS test severity
    ks_severity = min(report['ks_p_value'] / 0.05, 1.0)
    scores.append(1 - ks_severity)  # Lower p_value = higher severity

    # PSI severity
    psi_severity = min(report['psi'] / 0.25, 1.0)
    scores.append(psi_severity)

    # MMD severity
    mmd_severity = min(report['mmd'] / report['mmd_threshold'], 1.0)
    scores.append(mmd_severity)

    # Weighted average
    severity = (scores[0] * 0.3 + scores[1] * 0.4 + scores[2] * 0.3)
    return severity
```

### Seasonal Drift

Handle expected seasonal variations:

```python
# Different reference data for each season
reference_data_summer = load_reference('summer')
reference_data_winter = load_reference('winter')

current_season = get_current_season()
reference = reference_data_summer if current_season == 'summer' else reference_data_winter

detector = DataDriftDetector(reference_data=reference)
drift_detected = detector.detect_drift(current_data)
```

## Troubleshooting

### False Positives (Detecting drift when there isn't any)

**Symptoms**: Frequent drift alerts, but model performs fine

**Solutions**:
1. Increase p-value threshold: `p_value_threshold=0.01`
2. Use stricter consensus: require all 3 tests to agree
3. Increase reference data size for better calibration

### False Negatives (Missing actual drift)

**Symptoms**: Model performance degrades but no drift detected

**Solutions**:
1. Decrease p-value threshold: `p_value_threshold=0.1`
2. Use more sensitive test: focus on PSI
3. Check if reference data is representative

### Slow Detection

**Problem**: Tests take too long to run

**Solutions**:
1. Use only fast test: `test_types=['ks', 'psi']`
2. Subsample data: `detector.detect_drift(sample(data, 1000))`
3. Use approximate MMD implementation

---

**Next Steps**:
- Setup [Production Monitoring](deployment.md#monitoring--logging)
- Create [Retraining Pipeline](training.md#advanced-training)
- Monitor via [Dashboard](deployment.md#metrics-collection)
