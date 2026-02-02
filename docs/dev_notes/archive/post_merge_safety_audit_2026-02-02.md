# Post-Merge Safety Audit — React Consolidation

**Branch:** `post-merge-safety-audit`  
**Date:** 2026-02-02  
**Scope:** Post-merge safety pass after consolidating two React apps into a single SPA; CI passed but BugBot review was skipped.

---

## What was audited

- **React consolidation**
  - Confirmed exactly one React app build pipeline: `scripts/build_dashboard.py` (builds `_vendor/figma_make`, copies to `app/static/dashboard/`). Single `package.json`, single Vite config with `base: '/static/dashboard/'`.
  - Searched for leftover references to removed dashboards/builds: no code or CI references to `build_model_info_dashboard.py` or `app/static/model_info_dashboard/`; only historical mentions in `docs/dev_notes/2026-02-02.md`.
  - React Router: `App.tsx` uses `BrowserRouter` with routes `/api/dashboard`, `/api/model-info-dashboard`, and `*`. Client-side navigation and hard refresh on non-root routes are supported: Flask serves `index.html` for `/api/dashboard` and `/api/dashboard/<path:path>` (and same for `/api/model-info-dashboard`), so SPA fallback is correct.

- **Flask + static serving**
  - `/api/dashboard` and `/api/dashboard/<path:path>` serve `app/static/dashboard/index.html` via `send_from_directory(current_app.static_folder, "dashboard/index.html")`. Same for `/api/model-info-dashboard` and `/api/model-info-dashboard/<path:path>`.
  - Static assets: built `index.html` references `/static/dashboard/assets/...`; Flask default static route serves from `app/static`, so `/static/dashboard/` is correct.
  - Assumption: dashboard is **prebuilt**; CI (`scripts/ci.sh`) does **not** run `build_dashboard.py` — it only runs pytest. Tests that hit the SPA require `app/static/dashboard/` to exist from a prior build.

- **Existing guardrails**
  - `test_api_contract_stubs.py`: `test_api_responses_contain_no_nan_or_infinity` includes GET `/api/model-info/dashboard`; `_json_contains_no_nan_or_infinity` is recursive (dict/list/float) and covers nested fields. `test_api_model_info_dashboard_contract` and `test_model_info_dashboard_null_realism` assert contract shape and no NaN with empty model dir.
  - API: `_build_model_info_dashboard_payload()` uses `_safe_float_prob`, `_safe_category_display`, and explicit NaN/Inf checks for `last_updated`, support, precision, recall. No additional runtime JSON sanitizer added.

---

## Risks found

- **Prebuilt assets in CI:** CI does not build the dashboard. If `app/static/dashboard/` is missing or stale, SPA routes still return 200 as long as `index.html` exists (e.g. from a previous local build). New smoke tests will fail if the file is missing, but CI does not enforce a fresh build — documented assumption only.
- **No other functional issues:** Route registration, SPA fallback, and model-info API behavior are consistent with a single SPA and one build pipeline.

---

## Guardrails added

- **Smoke tests** in `tests/test_app_smoke.py`:
  - `test_dashboard_spa_returns_200_and_references_static_assets`: GET `/api/dashboard` returns 200, `text/html`, and response body contains `/static/dashboard/`.
  - `test_model_info_dashboard_spa_returns_200`: GET `/api/model-info-dashboard` returns 200 and HTML that references `/static/dashboard/`.
  - `test_model_info_dashboard_api_returns_200_and_valid_json`: GET `/api/model-info/dashboard` returns 200 and JSON with keys `model`, `metrics`, `categories`, `criticalThresholds`, `registry`.

- **NaN/Infinity:** Existing recursive test helper and API-side safe helpers already cover nested fields; no new JSON validation helper added.

---

## Explicitly left untouched

- No refactors of React components, route names, or data contracts.
- No changes to `run.py` dashboard build check, `scripts/build_dashboard.py`, or Vite/React Router config.
- No new CI step to run `build_dashboard.py` (would require Node in CI; left as prebuilt assumption).
- No runtime JSON sanitizer in app code; reliance on existing payload construction and contract tests only.
