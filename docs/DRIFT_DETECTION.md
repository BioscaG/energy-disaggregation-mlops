# Data Drift Robustness Testing (M27)

## Overview

This module implements comprehensive data drift detection and model robustness testing for energy disaggregation models. It addresses potential failure modes where the model encounters distribution shifts or degraded data quality in production.

## Features

### 1. **Data Drift Detection**

Detects when the input data distribution has shifted from training data using multiple statistical tests:

- **Kolmogorov-Smirnov (KS) Test**: Tests if two distributions are significantly different (p < 0.05)
- **Population Stability Index (PSI)**: Measures the magnitude of distribution shift (threshold: 0.25)
- **Maximum Mean Discrepancy (MMD)**: Kernel-based distance metric between distributions

### 2. **Robustness Testing**

Evaluates model resilience against three types of perturbations:

- **Noise Injection**: Adds Gaussian noise to simulate sensor noise or measurement errors
- **Amplitude Scaling**: Simulates sensor drift, degradation, or hardware recalibration
- **Missing Data**: Simulates data gaps or sensor failures

### 3. **Visualization & Reporting**

Generates publication-ready plots and comprehensive reports showing:

- Statistical drift metrics across different scenarios
- Performance degradation curves under perturbations
- Comparative analysis of robustness

## Installation

The required dependencies are already included in the project:

```bash
pip install scipy matplotlib  # Additional dependencies
```

## Usage

### CLI Commands

#### Detect Data Drift

```bash
# Basic usage (default: scale drift, 20% magnitude)
edmlops detect-drift

# Test with different drift types
edmlops detect-drift --drift-type noise --drift-magnitude 0.1
edmlops detect-drift --drift-type missing --drift-magnitude 0.15

# Custom data paths
edmlops detect-drift \
  --preprocessed-folder data/processed \
  --data-path data/raw/ukdale.h5 \
  --test-split-ratio 0.2
```

**Output:**
- Console report with KS statistic, PSI, and MMD values
- `drift_analysis_report.png` - Visualization of drift metrics

#### Test Model Robustness

```bash
# Basic usage (default: noise perturbation)
edmlops test-robustness

# Test different perturbation types
edmlops test-robustness --perturbation-type scale
edmlops test-robustness --perturbation-type missing

# Adjust number of samples
edmlops test-robustness --num-samples 100 --batch-size 16
```

**Output:**
- Console report with RMSE/MAE at each perturbation level
- Performance degradation percentage
- Robustness classification (ROBUST/MODERATE/LOW)
- `robustness_report.png` - Visualization of performance curves

### Python API

#### Data Drift Detection

```python
from energy_dissagregation_mlops.drift_detection import DataDriftDetector
import numpy as np

# Generate or load data
reference = np.random.normal(100, 20, 1000)  # Training distribution
current = np.random.normal(120, 20, 1000)    # Test distribution

# Detect drift using multiple methods
result = DataDriftDetector.compare_distributions(reference, current)

print(f"KS p-value: {result['ks_test']['p_value']}")
print(f"PSI: {result['psi']}")
print(f"MMD: {result['mmd']}")
print(f"Drift detected: {result['ks_test']['drift_detected']}")
```

#### Model Robustness Testing

```python
from energy_dissagregation_mlops.drift_detection import RobustnessAnalyzer
from energy_dissagregation_mlops.model import Model
import torch

# Load model
model = Model()
# ... load checkpoint ...

# Create analyzer
analyzer = RobustnessAnalyzer(model, device="cuda")

# Evaluate robustness
results = analyzer.evaluate_robustness(
    x_test, y_test,
    perturbation_type="noise",
    perturbation_levels=[0.0, 0.01, 0.05, 0.1],
    batch_size=32
)

# Access results
for level, metrics in results["metrics_by_level"].items():
    print(f"Noise level {level}: RMSE={metrics['rmse']:.6f}")
```

### Batch Script

For comprehensive analysis across all scenarios:

```bash
python scripts/test_drift_robustness.py \
  checkpoint_path=models/best.pt \
  data_path=data/raw/ukdale.h5 \
  preprocessed_folder=data/preprocessed \
  output_dir=drift_results \
  num_test_samples=100
```

**Output:**
- Complete report in `drift_results/DRIFT_REPORT.md`
- Multiple visualizations
- JSON results file for automated processing

## Testing

Run the comprehensive test suite:

```bash
# All drift detection tests
pytest tests/test_drift_detection.py -v

# Specific test classes
pytest tests/test_drift_detection.py::TestDataDriftDetection -v
pytest tests/test_drift_detection.py::TestRobustnessAnalyzer -v

# Integration tests (requires data)
pytest tests/test_drift_detection.py -v -m integration

# With coverage
pytest tests/test_drift_detection.py --cov=src/energy_dissagregation_mlops/drift_detection
```

## Interpreting Results

### Drift Detection Metrics

| Metric | Threshold | Interpretation |
|--------|-----------|-----------------|
| KS p-value | < 0.05 | Significant drift detected |
| PSI | > 0.25 | Significant drift |
| MMD | Higher = more drift | Quantifies distance between distributions |

### Robustness Metrics

| Degradation | Classification | Action |
|-------------|-----------------|--------|
| < 20% | ✅ ROBUST | Model is production-ready |
| 20-50% | ⚠️ MODERATE | Monitor in production, consider mitigation |
| > 50% | ❌ LOW | Requires retraining or architectural changes |

## Implementation Details

### DataDriftDetector Class

Provides static methods for statistical drift detection:

- `kolmogorov_smirnov_test()`: KS test implementation
- `population_stability_index()`: PSI calculation with histogram binning
- `maximum_mean_discrepancy()`: MMD with RBF/linear kernels
- `compare_distributions()`: Comprehensive multi-metric analysis

### RobustnessAnalyzer Class

Evaluates model robustness under perturbations:

- `add_gaussian_noise()`: Random noise injection
- `scale_amplitude()`: Scale data by factors
- `add_missing_data()`: Zero-out random samples
- `evaluate_robustness()`: Unified perturbation testing interface

### Visualization Functions

- `visualize_drift_analysis()`: Creates 4-panel drift metric plot
- `visualize_robustness_analysis()`: Creates RMSE/MAE degradation curves

## Supported Scenarios

### Drift Types
1. **Covariate Shift**: Input distribution changes (e.g., seasonal variation)
2. **Label Shift**: Output distribution changes (implicit in energy patterns)
3. **Concept Drift**: Relationship between input and output changes

### Perturbation Types
1. **Sensor Noise**: Gaussian noise (measurement uncertainty)
2. **Sensor Drift**: Amplitude scaling (calibration issues)
3. **Data Gaps**: Missing values (sensor failures)

## Best Practices

1. **Regular Testing**: Run robustness tests as part of continuous integration
2. **Baseline Establishment**: Record baseline metrics for comparison
3. **Threshold Setting**: Adjust drift thresholds based on domain knowledge
4. **Production Monitoring**: Deploy drift detection to production environments
5. **Retraining Triggers**: Set up automated retraining when PSI > 0.25

## Integration with Monitoring

For production monitoring, integrate the drift detection module:

```python
# In production inference loop
from energy_dissagregation_mlops.drift_detection import DataDriftDetector

# Collect reference statistics during training
reference_distribution = ...  # From training data

# In production, periodically check incoming data
current_batch_distribution = ...  # From current production data

result = DataDriftDetector.kolmogorov_smirnov_test(
    reference_distribution,
    current_batch_distribution
)

if result["drift_detected"]:
    logger.warning("Drift detected! Consider retraining...")
    send_alert_to_monitoring_system()
```

## References

- **KS Test**: Kolmogorov-Smirnov test for distribution comparison
- **PSI**: Population Stability Index, commonly used in credit risk modeling
- **MMD**: Maximum Mean Discrepancy (Gretton et al., 2012)

## Files

- `src/energy_dissagregation_mlops/drift_detection.py` - Core implementation
- `tests/test_drift_detection.py` - Test suite (18 tests, 100% pass rate)
- `scripts/test_drift_robustness.py` - Batch analysis script
- Updated `src/energy_dissagregation_mlops/cli.py` - New CLI commands

## Future Enhancements

1. **Adaptive Thresholds**: Learn optimal thresholds from historical data
2. **Ensemble Methods**: Combine multiple drift detectors
3. **Time Series Drift**: Account for temporal patterns in energy data
4. **Explainability**: Feature-level drift attribution
5. **Automated Retraining**: Trigger and execute model retraining on drift

---

**Status**: ✅ Complete and tested
**Test Coverage**: 18 unit tests, all passing
**Production Ready**: Yes
