# Test FAQ

## Why do some tests accept multiple status codes (200/302/400)?
The prediction route can redirect or return a validation error depending on the active configuration. For example, CSRF-enabled environments may redirect back to `/` when a token is missing, while the test configuration returns `200` with mock predictions. Allowing the short list of expected statuses demonstrates resilience without masking real failures.

## Why do certain tests skip when the model is missing?
Production-config tests and the performance suite require the trained pickle on disk. Rather than failing noisily in CI, the helper `skip_if_no_model` reports the missing dependency with a consistent reason. This keeps the fast path deterministic but reminds maintainers to supply the artifact when running extended checks.

## How do I regenerate the compare-model artifacts?
The compare-model tests look for dated experiment folders under `experiments/`. To refresh them, run:

```bash
python scripts/compare_models.py
```

This command updates the experimental run directories and `experiments/results/performance_metrics.csv`. The tests operate on temporary copies, so there is no need to commit the generated CSVs.

## What if NLTK downloads hang during testing?
The test suite never triggers live downloads; `tests/test_optimization.py` explicitly asserts that `nltk.download` is absent from `app/config.py`. If you hit a hang when running the application itself, pre-download the corpora by executing `python app/nltk_setup.py` or running the app once with network access. The tests will continue to pass because they rely on mocks rather than runtime downloads.

