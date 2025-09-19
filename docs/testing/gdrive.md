# Google Drive Testing

The Google Drive suite documents and verifies the contract for downloading models via `ModelService`. Every test is fully mocked to keep CI hermetic while still exercising error handling and cleanup paths.

## Contract

- **URL shape** – Requests are sent to `https://drive.google.com/uc?export=download&id=<FILE_ID>`.
- **Headers** – A binary response (`content-type: application/octet-stream`) indicates success; HTML responses trigger a hard failure because Google often uses HTML for warnings or auth challenges.
- **File hygiene** – Temporary files created during download must be removed whether the call succeeds or fails.
- **Model validation** – After download, `joblib.load` should succeed and the resulting model must answer `predict`/`predict_proba` calls.

## How the tests work

`tests/test_gdrive_deployment.py` patches `requests.get` for each scenario. The helper `_configure_mock_response` builds a context manager that mimics streaming responses so the production code path is exercised without real network access. `joblib.load` is also patched where necessary to avoid deserialising large artifacts.

The suite covers:

- Placeholder ID rejection and environment variable handling.
- Successful download and prediction flow (with mock payloads).
- HTML, network, timeout, corrupted, and undersized file errors.
- Cleanup guarantees – every test asserts that no `.tmp` files remain in the download directory after execution.

## When to run against the real API

For most work you can rely on the mocked tests. To validate the real integration:

1. Export a valid `GDRIVE_MODEL_ID` that points to a production model in your Drive.
2. Ensure the model is shared appropriately so the CI agent (or your local account) can access it.
3. Run the opt-in test: `pytest -q tests/test_gdrive_deployment.py -k real_id`.

This path is skipped by default and should be scheduled sparingly (e.g., nightly or pre-release) because it will perform an actual download and depends on external availability.

