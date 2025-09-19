# Performance Testing

The performance suite focuses on the time it takes to warm and reload the production model. The goal is to guarantee that the Flask worker can recycle without incurring costly cold starts.

## What is measured?

- **Reload time** – `tests/test_perf.py` loads the model once to populate the cache, reloads immediately, and asserts the second call finishes in under **150 ms**.
- **Logging** – The test writes the measured time to the `tests.perf` logger so slow runs leave breadcrumbs in CI logs.

## Preparing your environment

1. Ensure the production pickle exists at `model/disaster_rf_v1-2-0_prod_2025-09-11.pkl` (or update `Config.MODEL_PATH`).
2. If you rely on Google Drive, download the model once before executing the perf suite so the file is present locally.
3. Activate the virtual environment and install requirements so `ModelService` can import dependencies.

```bash
pytest -q -m perf
```

## Interpreting results

- **Pass** – Reload completes under the threshold. Keep the printed duration in the logs for historical tracking.
- **Fail** – The assertion message includes the observed duration. Investigate CPU contention, missing optimisations, or incompatible models.
- **Skipped** – `skip_if_no_model` reports when the artifact is missing. Provide the pickle or run the Google Drive workflow before re-running.

Because timing on developer machines can be noisy, consider:

- Closing background tasks that compete for CPU.
- Running the perf suite multiple times and comparing the logs to confirm a trend.
- Adjusting the threshold only after profiling the bottleneck and documenting the change.

The perf marker is excluded by default to keep pull-request CI fast, but schedule it for nightly or pre-release jobs to avoid regressions.

