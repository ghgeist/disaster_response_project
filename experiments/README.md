# Experiments

This folder contains experimental model runs, configurations, and comparison results.

## Folder Structure

- **`experimental_runs/{YYYY-MM-DD}/`** - Dated experiment results. Each folder contains models, metrics, logs, and reports from that date's experiments. Self-contained and organized by date.

- **`model_candidates/`** - Hyperparameter configurations that have been tested or are candidates for testing.

- **`experimental_configs/`** - Reusable experiment configurations (hyperparameters, eval sets, sampling strategies).

- **`comparisons/`** - Timestamped model comparison reports showing performance differences between models.

- **`model_archive/`** - Archived production models and their metadata.

- **`logs/`** - Training and execution logs from experimental runs.

- **`results/`** - Legacy folder (currently empty).

## Conventions

- Each dated folder in `experimental_runs/` is self-contained with all artifacts from that experiment date.
- Look for `*_comparison_report.md` or `vocabulary_comparison_report.md` files in dated folders for detailed experiment summaries.
- Model metadata is stored in `MODEL_INFO.json` files within each experiment run.

