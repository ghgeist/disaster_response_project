Sampling strategy JSON format

Purpose
- Define experiment presets for handling class imbalance that the script auto-discovers and lists in its menu.

Discovery rules
- Directory: experiments/experimental_configs/sampling_strategies
- Loads: all files ending with .json
- Ignores: files starting with _ and files ending with .disabled.json
- Override path: CLI flag --strategies-dir or env STRATEGIES_DIR

Minimal required fields
- config.sampling_method: one of baseline, smote, adasyn, conservative

Optional fields
- experiment_name: unique identifier for the run preset
- display_name: friendly label shown in the menu
- description: short explanation shown in the menu
- order: integer; lower shows first. If omitted, entries are sorted by display_name or filename
- config.test_size: float in [0, 1] (default: 0.2 if omitted)
- config.random_state: integer (default: 42 if omitted)

Disable a preset
- Rename file to *.disabled.json (or prefix file with _)

Examples

Minimal
{
  "experiment_name": "baseline_no_sampling_v1",
  "config": {
    "sampling_method": "baseline"
  }
}

With menu metadata and custom order
{
  "experiment_name": "smote_conservative_v1",
  "display_name": "SMOTE (Conservative)",
  "description": "SMOTE with conservative parameters",
  "order": 20,
  "config": {
    "sampling_method": "smote",
    "test_size": 0.2,
    "random_state": 42
  }
}

Validation (optional)
- A JSON Schema is provided in schema.json. Validation is optional at runtime.
- To validate with Python (requires jsonschema):
  python - << 'PY'
  import json, sys
  from jsonschema import validate, Draft7Validator
  import pathlib
  schema = json.load(open('experiments/experimental_configs/sampling_strategies/schema.json'))
  ok = True
  for p in pathlib.Path('experiments/experimental_configs/sampling_strategies').glob('*.json'):
      if p.name.endswith('.disabled.json') or p.name.startswith('_'):
          continue
      try:
          data = json.load(open(p))
          Draft7Validator.check_schema(schema)
          validate(data, schema)
          print(f"OK  {p}")
      except Exception as e:
          ok = False
          print(f"ERR {p}: {e}")
  sys.exit(0 if ok else 1)
  PY


