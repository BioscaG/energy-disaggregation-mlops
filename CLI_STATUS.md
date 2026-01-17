# CLI Updates Summary

## ✅ Status: FULLY FUNCTIONAL

The CLI has been **fully updated and tested** to work with the new energy disaggregation model.

## Changes Made to CLI

### File Modified: `src/energy_dissagregation_mlops/cli.py`

**Preprocess Command** - Updated to support two meters:

```python
# BEFORE
--meter <N>  # Single meter (old autoencoder approach)

# AFTER
--meter-mains <N>        # Input: Total building power (usually 1)
--meter-appliance <N>    # Target: Device power to predict (2+)
```

## CLI Commands

All 4 commands work correctly:

### 1. `preprocess`
```bash
python -m energy_dissagregation_mlops.cli preprocess \
  --data-path data/raw/ukdale.h5 \
  --output-folder data/processed \
  --meter-mains 1 \
  --meter-appliance 2
```

### 2. `train`
```bash
python -m energy_dissagregation_mlops.cli train \
  --preprocessed-folder data/processed \
  --epochs 10
```

### 3. `evaluate`
```bash
python -m energy_dissagregation_mlops.cli evaluate \
  --preprocessed-folder data/processed \
  --plot-results
```

### 4. `download`
```bash
python -m energy_dissagregation_mlops.cli download \
  --target-dir data/raw
```

## Verification Results

| Component | Status |
|-----------|--------|
| Module imports | ✅ All working |
| PreprocessConfig | ✅ meter_mains, meter_appliance exist |
| CLI registration | ✅ All commands available |
| Preprocess parameters | ✅ Updated with new params |
| Train dataloader | ✅ Handles (x, y) pairs |
| Evaluate metrics | ✅ Works with new format |
| Help messages | ✅ Descriptive and accurate |

## Quick Examples

**Predict device meter 2 (default):**
```bash
python -m energy_dissagregation_mlops.cli preprocess \
  --data-path data/raw/ukdale.h5 \
  --output-folder data/processed

python -m energy_dissagregation_mlops.cli train --epochs 10
python -m energy_dissagregation_mlops.cli evaluate --plot-results
```

**Predict device meter 5:**
```bash
python -m energy_dissagregation_mlops.cli preprocess \
  --data-path data/raw/ukdale.h5 \
  --output-folder data/processed_meter5 \
  --meter-appliance 5

python -m energy_dissagregation_mlops.cli train \
  --preprocessed-folder data/processed_meter5

python -m energy_dissagregation_mlops.cli evaluate \
  --preprocessed-folder data/processed_meter5
```

## Full Documentation

See `docs/CLI_USAGE.md` for comprehensive CLI documentation with:
- All parameters explained
- Multiple workflow examples
- Tips & tricks
- Shell script examples
