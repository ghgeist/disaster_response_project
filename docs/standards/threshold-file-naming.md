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
    "critical_target_recall": 0.65,
    "non_critical_target_recall": 0.60,
    "optimization_method": "precision_recall_curve",
    "optimization_split": "calibration",
    "reporting_split": "frozen_eval",
    "calibration_ids": "experiments/experimental_configs/eval_sets/cal_ids.json",
    "eval_ids": "experiments/experimental_configs/eval_sets/eval_ids.json"
  },
  "thresholds": {
    "medical_help": 0.124,
    "water": 0.362,
    "food": 0.431
  },
  "calibration_stats": [
    {
      "category": "medical_help",
      "type": "critical",
      "threshold": 0.124,
      "actual_recall": 0.65,
      "note": "diagnostic only — measured on cal"
    }
  ],
  "category_stats": [
    {
      "category": "medical_help",
      "type": "critical",
      "threshold": 0.124,
      "actual_recall": 0.61,
      "note": "reported performance — measured on frozen eval"
    }
  ],
  "performance": {
    "baseline": {"f1_weighted": 0.93, "note": "eval"},
    "optimized": {"f1_weighted": 0.90, "critical_recall": 0.61, "note": "eval"},
    "delta": {"f1_weighted": -0.03}
  }
}
```

`category_stats` and `performance` are always **frozen-eval** metrics (model-info / reporting semantics). `calibration_stats` is diagnostic-only and must not replace `category_stats` for dashboards.
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

- `scripts/03_optimization/optimize_per_category_thresholds.py`: Saves **canonical** `{model_stem}_thresholds.json` (also saves legacy name). Tunes on cal; `category_stats` / `performance` are frozen-eval.
- `scripts/02_training/03_create_experimental_model.py`: Saves diagnostic `{model_stem}_f2_thresholds.json` only (does **not** overwrite the canonical thresholds file). Under frozen three-way, F2 tunes on cal.

### Training Scripts

- `scripts/03_create_experimental_model.py`: Saves `{model_stem}_f2_thresholds.json` in experiment directory
- `scripts/04_create_production_model.py`: Saves as `thresholds.json` (legacy RF path; follow-up)

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
- `scripts/optimize_per_category_thresholds.py` - Per-category threshold optimization
- `scripts/model_naming_utility.py` - Model and artifact renaming utility

