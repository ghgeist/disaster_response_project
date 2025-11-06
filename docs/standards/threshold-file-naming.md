# Threshold File Naming Standard

## Standard Format

**Primary naming convention**: `{model_stem}_thresholds.json`

Where `{model_stem}` is the model filename without the `.pkl` extension.

### Examples

- Model: `disaster_lr_vocab15k_prod_2025-11-06.pkl`
- Thresholds: `disaster_lr_vocab15k_prod_2025-11-06_thresholds.json`

- Model: `lr_baseline_model.pkl`
- Thresholds: `lr_baseline_model_thresholds.json`

## File Location

Threshold files should be co-located with the model file in the same directory:

```
model/
  ├── disaster_lr_vocab15k_prod_2025-11-06.pkl
  └── disaster_lr_vocab15k_prod_2025-11-06_thresholds.json
```

## File Format

### Standard Structure (Nested)

```json
{
  "metadata": {
    "created": "2025-11-04T21:39:20.384846",
    "model": "path/to/model.pkl",
    "target_recall": 0.65,
    "optimization_method": "precision_recall_curve"
  },
  "thresholds": {
    "medical_help": 0.124,
    "water": 0.362,
    "food": 0.431,
    ...
  },
  "critical_only": {
    "medical_help": 0.124,
    "water": 0.362,
    ...
  },
  "performance": {
    "baseline": {...},
    "optimized": {...},
    "delta": {...}
  }
}
```

### Legacy Structure (Flat)

```json
{
  "medical_help": 0.124,
  "water": 0.362,
  "food": 0.431,
  ...
}
```

## Loading Priority

The production app (`app/services.py`) loads thresholds in this priority order:

1. `{model_stem}_thresholds.json` - **Standardized (preferred)**
2. `optimized_critical_thresholds.json` - Legacy: optimized critical thresholds
3. `optimized_all_thresholds.json` - Legacy: optimized all thresholds
4. `thresholds.json` - Legacy: F2-optimized thresholds

## Script Behavior

### Optimization Scripts

- `scripts/optimize_critical_thresholds_inc1.py`: Saves as `{model_stem}_thresholds.json` (also saves legacy name)
- `scripts/optimize_all_thresholds.py`: Saves as `{model_stem}_thresholds.json` (also saves legacy name)

### Training Scripts

- `scripts/03_create_experimental_model.py`: Saves as `{model_stem}_thresholds.json` in experiment directory (if model name known)
- `scripts/04_create_production_model.py`: Saves as `thresholds.json` (legacy, should be updated)

## Migration Guide

### For Existing Models

1. **Rename existing threshold files**:
   ```powershell
   # Example
   mv model/optimized_critical_thresholds.json model/disaster_lr_vocab15k_prod_2025-11-06_thresholds.json
   ```

2. **Or keep both** (app will prefer standard name):
   - Standard name: `{model_stem}_thresholds.json`
   - Legacy name: `optimized_critical_thresholds.json` (for backward compatibility)

### For New Models

Always use the standard naming: `{model_stem}_thresholds.json`

## Benefits

1. **Clear association**: Threshold file name matches model file name
2. **No ambiguity**: Easy to identify which thresholds belong to which model
3. **Tool compatibility**: Works with `model_naming_utility.py` for renaming
4. **Backward compatible**: App still loads legacy names as fallback

## Related Files

- `app/services.py` - Threshold loading logic
- `scripts/optimize_critical_thresholds_inc1.py` - Critical threshold optimization
- `scripts/optimize_all_thresholds.py` - All-category threshold optimization
- `scripts/model_naming_utility.py` - Model and artifact renaming utility

