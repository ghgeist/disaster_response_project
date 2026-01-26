# disasterproject package

Core reusable Python package for the disaster response classification project.
It contains the data pipeline, model training and evaluation components, and
shared utilities used by both the Flask app and the training scripts.

## Package layout

```
disasterproject/
|-- data/
|-- models/
|-- evaluation/
|-- utils/
|-- hierarchy.py
`-- __init__.py
```

## Subpackages

- data/: ETL pipeline, dataset schemas, and feature preparation helpers.
- models/: Training workflows, model configuration, and persistence utilities.
- evaluation/: Metrics, reporting, and model comparison helpers.
- utils/: Shared configuration and I/O helpers.
- hierarchy.py: Hierarchy constraints and post-processing logic for labels.

## Role in the project

This package is installed with `pip install -e .` so it can be imported from
the Flask application and the reproducible training scripts.

- app/ uses it to load models, apply preprocessing, and enforce hierarchy
  constraints before returning predictions to users.
- scripts/ uses it for ETL, training, evaluation, and experiment workflows.

## Usage

```
pip install -e .
```

```
from disasterproject.data import ...
from disasterproject.models import ...
```

## Notes

- Keep the package focused on reusable logic and avoid app-specific concerns.
- Public helpers should remain small, single-purpose, and well documented.

## Related docs

- scripts/README.md for workflow-specific guidance
- app/ for the Flask web UI entry point
