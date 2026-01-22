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
