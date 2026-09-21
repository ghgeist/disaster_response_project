# Google Drive Testing (historical)

> **Status (2026-09-21):** Runtime Google Drive downloads are **retired**. The Flask `ModelLoader` loads only local `model/*.pkl` artifacts (auto-discovered `disaster_*_prod_*.pkl`, optional `MODEL_FILENAME` override). This page is retained so the former contract is not erased; do not treat it as an active deployment path.

## Historical contract

The previous Drive suite documented downloading models via a `ModelService` helper that no longer ships in the app loader:

- **URL shape** – Requests were sent to `https://drive.google.com/uc?export=download&id=<FILE_ID>`.
- **Headers** – A binary response (`content-type: application/octet-stream`) indicated success; HTML responses failed closed.
- **File hygiene** – Temporary download files were removed on success or failure.
- **Model validation** – After download, `joblib.load` had to succeed and answer `predict` / `predict_proba`.

## Historical tests

`tests/test_gdrive_deployment.py` (removed / not present on current main) used to patch `requests.get` and `joblib.load` so CI stayed hermetic while exercising error handling. Markers such as `gdrive` may still appear in older docs or configs; they are not required for current local-artifact deploys.

## Current replacement checks

Prefer these for active work:

```bash
# Stem-bound companions + thresholds SHA provenance
python scripts/run_tests.py tests/test_thresholds_alignment.py -q

# App smoke / model health with local pickle
python scripts/run_tests.py tests/test_app_smoke.py -q
```

See [deployment runbook](../runbooks/deployment.md) and [ADR-003](../adr/adr-003-hybrid-model-deployment-strategy.md) for the local-artifact strategy and labeled GDrive history.
