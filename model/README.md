# Model Artifacts

This folder stores model files and their companion metadata used by the app and
evaluation scripts.

## Naming
- Preferred thresholds filename: `{model_stem}_thresholds.json`
- Legacy fallback filename: `optimized_critical_thresholds.json`

## Notes
- Keep model files and matching thresholds in this folder together.
- The app looks for thresholds alongside the model and will prefer the
  standardized filename when present.

## Script Dependencies (2026-01-22)

**Note**: `model/parameters.json` and `model/class_weights.json` were removed on 2026-01-22.

- **Reason**: These files were defaults for `scripts/04_create_production_model.py` (RandomForest only), but the current production model is LogisticRegression created via `scripts/03_create_experimental_model.py`.
- **Impact**: If you need to run `scripts/04_create_production_model.py`, you must provide `--params` and `--class-weights` arguments pointing to appropriate config files.
- **Current workflow**: Use `scripts/03_create_experimental_model.py` with configs from `experiments/model_candidates/` (e.g., `vocab_15k.json`), then promote via `scripts/promote_model.py`.
