---
created: 2026-01-29
updated: 2026-01-29
status: active
---

# Storm Signal Dashboard — Design Implementation Plan

> **Goal**: Adjust Storm Signal implementation plan to match existing repo state with minimal churn, avoiding CI hell and regression risks.

> **Status**: PLANNING PHASE — React/Vite/Tailwind not installed (intentional)

---

## A) Repo Inventory

### Backend Entrypoints & Routes

**Primary Entrypoint**: `run.py`
- Creates Flask app via factory pattern (`app.app.create_app`)
- Handles Replit deployment (reads `$PORT` env var)
- Runs on `0.0.0.0:$PORT` (default 5000)

**WSGI Entrypoint**: `wsgi.py`
- Gunicorn-compatible entry point for production
- Used by `.replit` deployment config

**Route Blueprints** (`app/routes/`):
- `home_bp`: `/`, `/index`, `/favicon.ico` (Jinja templates)
- `classification_bp`: `/go`, `/classify` (supports JSON via `format=json` or `Content-Type: application/json`)
- `health_bp`: `/health`, `/health/detailed`, `/api/model-health`, `/api/performance-diagnostics`

**Existing API Endpoints**:
- `POST /classify` (or `GET /classify?query=...&format=json`) — Returns dict (not JSON-ified) when `_is_json_request()` is True
  - Response shape: `{query, use_hierarchy, raw: {predictions, probabilities, labels}, fixed: {...}, violations, metrics}`
- `GET /health` — Returns `{'status': 'ok'}` (200)
- `GET /health/detailed` — Returns health diagnostics JSON
- `GET /api/model-health` — Returns model health report JSON
- `GET /api/performance-diagnostics` — Returns performance diagnostics JSON

**Missing API Endpoints** (per spec):
- `GET /api/feed` — Not implemented
- `GET /api/metrics` — Not implemented
- `GET /api/categories` — Not implemented

### Current UI Approach

**Template Engine**: Jinja2 (Flask default)
- Base template: `app/templates/base.html`
- Main page: `app/templates/home.html` (uses Tailwind CSS via CDN)
- Results page: `app/templates/results.html`
- Static assets: `app/static/css/` (includes `tailwind.css`)

**Current Frontend Stack**:
- Tailwind CSS loaded via CDN (not build-time)
- Plotly.js for visualizations (CDN)
- Vanilla JavaScript (no React/Vite)
- Forms use Flask-WTF with CSRF protection

**What Will Be Replaced**:
- Jinja templates → React components (Phase 2)
- CDN Tailwind → Build-time Tailwind via Vite (Phase 2)
- Form-based navigation → SPA routing (Phase 2)

### Testing Setup

**Test Framework**: pytest
- Config: `pytest.ini` (excludes `scripts/`, defines markers: `perf`, `integration`, `security`, `slow`)
- Fixtures: `tests/conftest.py` (provides `app`, `client` fixtures)
- Test structure: `tests/` directory with multiple test files

**Existing Test Files**:
- `test_app_smoke.py` — Smoke tests for app startup and prediction flow
- `test_flask_standardized.py` — Flask app configuration and model validation tests
- `test_smoke.py` — Additional smoke tests
- `test_hierarchy.py` — Hierarchy processing tests
- `test_security.py` — Security validation tests
- Others: `test_metrics_io.py`, `test_optimization.py`, `test_perf.py`, etc.

**Test Execution**:
- Portable runner: `scripts/run_tests.py` (handles Python executable detection)
- Standard: `pytest tests/test_smoke.py -q`
- Pre-commit hooks: `.pre-commit-config.yaml` (Black + Ruff)

**CI/CD**:
- No GitHub Actions workflows found (`.github/workflows/` empty)
- Pre-commit hooks exist but no CI pipeline

### Deployment Assumptions

**Replit Configuration** (`.replit`):
- Build: `pip install -r requirements.txt && python -m nltk.downloader ...`
- Run: `gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 120 wsgi:application`
- Deployment target: `autoscale`
- Port mapping: local 5000 → external 80

**Static Serving**:
- Flask serves static files from `app/static/` (via Flask default)
- No build step for frontend assets (CDN-based)

**Model Loading**:
- Models loaded from local files only (`model/` directory)
- Model path: `model/disaster_rf_prod_*.pkl` (auto-discovered latest)
- Google Drive download functionality was removed (2026-01-26); code may still exist but is not used

### Services & Data Layer

**Services** (`app/services/`):
- `model_service.py` — Model loading and prediction
- `data_service.py` — Database access (SQLite via SQLAlchemy)
- `category_mapper.py` — Category name mapping
- `health_service.py` — Health monitoring
- `metrics_service.py` — Metrics calculation
- Others: `artifact_loader.py`, `model_loader.py`, `model_predictor.py`, `threshold_manager.py`

**Database**:
- SQLite: `data/02_stg/stg_disaster_response.db`
- Table: `stg_disaster_response`
- Columns: `id`, `message`, `original`, `genre`, plus 36 category columns

**Data Access Pattern**:
- `DataService.get_data()` returns pandas DataFrame
- Cached in memory after first load

---

## B) Plan Diff

| Spec Item | Exists? | Notes | Action |
|-----------|---------|-------|--------|
| **API: POST /api/classify** | Partial | `/classify` exists, returns dict (not JSON-ified). Missing `/api/` prefix. Response shape differs (has `raw`/`fixed`/`violations`, missing `severity`). | Rename to `/api/classify`, add `jsonify()`, add `severity` calculation, standardize response shape |
| **API: GET /api/feed** | No | Not implemented | Create new endpoint |
| **API: GET /api/metrics** | No | Not implemented | Create new endpoint |
| **API: GET /api/categories** | No | Not implemented | Create new endpoint |
| **Severity Calculation** | No | Logic exists in spec but not in code | Implement `calculate_severity()` helper |
| **Category Mapping** | Partial | `category_mapper.py` exists but may not match spec groups | Verify/update mapping to match spec groups |
| **Simulated Timestamps** | No | No timestamp simulation logic | Add timestamp generation for feed items |
| **Frontend: React/Vite** | No | Jinja templates only | Phase 2 (explicit gate) |
| **Frontend: Tailwind Build** | No | CDN only | Phase 2 (explicit gate) |
| **Contract Tests** | Partial | Tests exist but no schema validation | Add JSON schema validation in Phase 0 |
| **Pre-commit Gate** | Partial | Pre-commit hooks exist (Black/Ruff) but no contract test gate | Add contract test command to pre-commit |

### Implicit Contracts (Currently Embedded in Templates)

**Critical**: The following contracts are currently implicit in Jinja templates and must become explicit API responses:

1. **Category Display Names**: Currently hardcoded in templates → Must be in `/api/categories` response
2. **Severity Badge Colors**: Currently CSS classes in templates → Must be in API response or frontend constants
3. **Feed Item Structure**: Currently rendered server-side → Must be JSON shape from `/api/feed`
4. **Metrics Structure**: Currently computed in `render_home_with_visualizations()` → Must be JSON shape from `/api/metrics`

---

## C) Adjusted Implementation Plan (Delta-Based)

### Phase 0: Planning + Contract Stabilization (No React Install)

**Objective**: Define and test API contracts before frontend changes.

#### Step 0.1: API Response Schema Definition
**Files**: `docs/api_contracts/`
- Create `api_contracts/classify_response.json` (JSON Schema)
- Create `api_contracts/feed_response.json`
- Create `api_contracts/metrics_response.json`
- Create `api_contracts/categories_response.json`
- Create `api_contracts/error_response.json` (standard error shape: `{success: false, error: str, code: int}`)

**Done means**: JSON Schema files exist and validate example responses.

#### Step 0.2: Severity Calculation Helper
**Files**: `app/utils/severity.py` (new)
- Implement `calculate_severity(categories: dict, probabilities: dict) -> str`
- Use spec algorithm (critical categories + confidence thresholds)
- Add unit tests: `tests/test_severity.py`

**Done means**: Function exists, tests pass, matches spec algorithm.

#### Step 0.3: Category Mapping Verification
**Files**: `app/services/category_mapper.py`
- Verify human-readable names match spec
- Verify category groups match spec (Critical Needs, Infrastructure, Weather Events, Other)
- Add `get_category_groups()` method if missing

**Done means**: Category mapping matches spec exactly.

#### Step 0.4: Enhance `/classify` → `/api/classify`
**Files**: `app/routes/classification.py`
- Keep existing `/classify` endpoint for backward compatibility (Jinja templates still use it)
- Add `/api/classify` route (reuse existing `_build_classify_response()` logic)
- Add `severity` field to response using new helper from Step 0.2
- Use `jsonify()` for proper JSON response
- **Response shape**: Keep existing structure, add `severity` field (minimal change approach)
  - Shape: `{query, use_hierarchy, raw: {...}, fixed: {...}, violations, metrics, severity: str}`
  - This maintains backward compatibility while adding the new field

**Done means**: `POST /api/classify` returns JSON with `severity` field, matches schema. `/classify` still works for templates.

#### Step 0.5: Create `/api/feed` Endpoint
**Files**: `app/routes/api.py` (new blueprint), `app/routes/__init__.py`
- Create `api_bp` blueprint
- Register `api_bp` in `app/routes/__init__.py` (add to `register_routes()` function)
- Implement `GET /api/feed`:
  - Query params: `limit` (default 25), `offset` (default 0), `categories[]` (filter)
  - Fetch messages from `DataService.get_data()` (pandas DataFrame)
  - Filter by category if `categories[]` provided (pandas filtering)
  - Generate simulated timestamps with "bursty" pattern:
    - Distribute over last 6 hours
    - Use weighted distribution (more recent = higher probability)
    - Add random bursts (clusters of messages within short time windows)
    - Most recent first (descending order)
    - Use deterministic seed in test mode (from Step 0.10)
  - Calculate severity for each item using helper from Step 0.2
  - Return JSON array of feed items matching schema

**Done means**: Endpoint exists, returns paginated feed items with bursty simulated timestamps. Blueprint registered.

#### Step 0.6: Create `/api/metrics` Endpoint
**Files**: `app/routes/api.py`
- Implement `GET /api/metrics`:
  - `vol_today`: Simulated count (can use database size as base)
  - `flagged_pct`: Simulated percentage
  - `top_categories`: Real counts from database using `DataService.get_data()` (top 5-7)
    - Use pandas DataFrame operations (no new DataService methods needed)
  - `trend_data`: Simulated time series (6 hours, hourly intervals)
    - Use bursty pattern similar to feed timestamps (clusters of activity)

**Done means**: Endpoint exists, returns metrics JSON matching schema. Uses existing DataService methods.

#### Step 0.7: Create `/api/categories` Endpoint
**Files**: `app/routes/api.py`
- Implement `GET /api/categories`:
  - Return all 36 categories with human-readable names (from `CategoryMapper`)
  - Return category groups mapping (from Step 0.3)
  - Return "Volume Today" counts (real from database using `DataService.get_data()`)
    - Use pandas DataFrame operations to count categories

**Done means**: Endpoint exists, returns category metadata matching schema. Uses existing DataService methods.

#### Step 0.8: Contract Tests + Schema Validation
**Files**: `tests/test_api_contracts.py` (new)
- Install `jsonschema` package (add to `requirements.txt`)
- Test each endpoint returns valid JSON matching schema
- Test response shapes match expected fields
- Test error responses (400, 500) return standard error shape: `{success: false, error: str, code: int}`
- Test edge cases (empty results, invalid params)

**Done means**: All API endpoints have contract tests that validate JSON Schema, including error responses.

#### Step 0.9: Pre-commit Gate
**Files**: `.pre-commit-config.yaml`
- Add hook: `python scripts/run_contract_tests.py` (new script)
- Script runs: `pytest tests/test_api_contracts.py -q`
- Must pass before commit

**Done means**: Pre-commit hook runs contract tests, blocks commits if tests fail.

#### Step 0.10: Deterministic Simulated Data (Test Mode)
**Files**: `app/utils/mocks.py` (enhance existing)
- Add `generate_deterministic_timestamps(count: int, seed: int = None, bursty: bool = True) -> list`
  - Generate timestamps over last 6 hours
  - If `bursty=True`: Use weighted distribution (more recent = higher probability) + random bursts (clusters)
  - If `bursty=False`: Uniform distribution
  - Most recent first (descending order)
  - Use seed for deterministic test behavior
- Add `generate_deterministic_metrics(seed: int = None) -> dict`
  - Generate metrics with bursty trend data (clusters of activity)
- Use in test mode for reproducible tests

**Done means**: Simulated data is deterministic in tests, reproducible across runs. Bursty pattern matches UI mockup.

---

**Phase 0 Gate**: All API endpoints exist, return JSON matching schemas, contract tests pass, pre-commit gate works.

---

### Phase 1: Tooling Probes (Optional)

**Objective**: Validate external tooling can read design files.

#### Step 1.1: Figma MCP Branch (Optional Spike)
**Files**: `docs/experiments/figma_mcp_spike.md`
- Test Cursor can read Figma design files via MCP
- Document findings
- Branch: `experiment/figma-mcp-validation`

**Done means**: Spike complete, findings documented, branch merged or abandoned.

**Phase 1 Gate**: Optional — skip if not needed.

---

### Phase 2: Frontend Setup (Explicitly Future)

**Objective**: Install React/Vite/Tailwind only after Phase 0 gates pass.

**Gate Condition**: Phase 0 complete + explicit approval to proceed.

#### Step 2.1: Frontend Initialization
**Files**: `frontend/` (new directory)
- Initialize Vite + React + TypeScript
- Install Tailwind CSS (build-time, not CDN)
- Install dependencies: `lucide-react`, `recharts`, `axios`, `react-resizable-panels`, `clsx`, `tailwind-merge`

**Done means**: `npm install` completes, `npm run dev` starts Vite dev server.

#### Step 2.2: Flask Integration
**Files**: `app/app.py`, `run.py`
- Configure Flask to serve `frontend/dist/index.html` for non-API routes
- Configure Vite proxy to Flask (`localhost:5000`) during dev
- Update `run.py` to handle both API and frontend serving

**Done means**: Flask serves React app, Vite proxies API requests, both work together.

#### Step 2.3: Minimal Vertical Slice
**Files**: `frontend/src/`
- Create `App.tsx` with header
- Create `ClassifyPanel.tsx` (right panel)
- Connect to `POST /api/classify`
- Render results (categories + severity)

**Done means**: Can classify a message in React UI, see results, app remains runnable.

**Phase 2 Gate**: React app loads, can classify messages, no regressions to existing Jinja routes.

---

## D) Minimal Test Strategy

### Pre-commit Gate (Primary Mechanism)

**Command**: `python scripts/run_contract_tests.py`

**Script Location**: `scripts/run_contract_tests.py` (new)

**What It Does**:
1. Runs `pytest tests/test_api_contracts.py -q`
2. Validates all API endpoints return JSON matching schemas
3. Exits with code 0 if pass, non-zero if fail

**Pre-commit Hook**:
```yaml
# .pre-commit-config.yaml addition
- repo: local
  hooks:
    - id: contract-tests
      name: API Contract Tests
      entry: python scripts/run_contract_tests.py
      language: system
      pass_filenames: false
      always_run: true
```

**Runtime Target**: < 5 seconds (fast enough for pre-commit)

### Test Structure (Reuse Existing)

**Framework**: pytest (existing)
**Fixtures**: Reuse `tests/conftest.py` (existing `app`, `client` fixtures)
**New Test File**: `tests/test_api_contracts.py`

**Test Coverage** (Phase 0):
- `test_api_classify_response_shape()` — Validates `/api/classify` response matches schema
- `test_api_feed_response_shape()` — Validates `/api/feed` response matches schema
- `test_api_metrics_response_shape()` — Validates `/api/metrics` response matches schema
- `test_api_categories_response_shape()` — Validates `/api/categories` response matches schema
- `test_severity_calculation()` — Unit tests for severity helper

**Schema Validation**: Use `jsonschema` library (add to `requirements.txt`)

### Frontend Tests (Phase 2, Future)

**When Introduced**: After React/Vite setup (Phase 2)

**Minimal Set**:
- Component rendering tests (Vitest + React Testing Library)
- API integration tests (mock `/api/classify` responses)
- No E2E tests initially (add later if needed)

**Test Command**: `npm test` (runs Vitest)

### CI Mirroring (Optional)

**If CI Added Later**:
- Mirror pre-commit gate: Run `pytest tests/test_api_contracts.py -q` on PR/push
- No nightly builds (per user requirement)
- Fast feedback (< 5 seconds)

---

## E) Risk Register

### Risk 1: API Contract Drift (Field Renames/Type Changes)

**Impact**: High — Frontend breaks silently if API shape changes

**Mitigation**:
- JSON Schema validation in contract tests (Phase 0.8)
- Pre-commit gate catches schema violations before commit
- Explicit schema files in `docs/api_contracts/` serve as source of truth
- Version API endpoints if breaking changes needed (`/api/v1/classify`)

**Owner**: Backend developer

---

### Risk 2: Hidden Coupling Between UI and Backend Formatting

**Impact**: High — Jinja templates contain formatting logic that must be replicated in React

**Mitigation**:
- Phase 0 extracts all formatting logic into explicit API responses
- Document implicit contracts (Section B above)
- Contract tests validate API responses, not template rendering
- Frontend uses API responses only (no server-side rendering in Phase 2)

**Owner**: Full-stack developer

---

### Risk 3: Silent Failure (Errors Swallowed, UI Appears "Fine")

**Impact**: High — Users see incorrect data without knowing

**Mitigation**:
- Contract tests validate error responses (400, 500) return proper JSON
- Add error response schemas (`api_contracts/error_response.json`)
- Frontend displays error states explicitly (Phase 2)
- Logging in backend (already exists via `app.logger`)

**Owner**: Backend + Frontend developer

---

### Risk 4: Simulated Data Inconsistency (Different Seeds in Dev vs Test)

**Impact**: Medium — Tests pass but dev behavior differs

**Mitigation**:
- Deterministic seed in test mode (Phase 0.10)
- Document seed values in test fixtures
- Use same seed for all simulated data in tests
- Allow seed override via env var for debugging

**Owner**: Backend developer

---

### Risk 5: Frontend Setup Breaks Existing Jinja Routes

**Impact**: Medium — App becomes unusable during migration

**Mitigation**:
- Phase 0 completes before React install (explicit gate)
- Flask serves both Jinja routes (`/`, `/go`) and React app (`/dashboard`) during transition
- Keep Jinja routes working until React app is fully functional
- Gradual migration: New features in React, old features remain in Jinja
- `/classify` endpoint remains for backward compatibility (Jinja templates)

**Owner**: Full-stack developer

---

### Risk 6: Performance Issues with Pandas DataFrame Operations

**Impact**: Low — App may be slower with large datasets

**Mitigation**:
- Use existing `DataService.get_data()` (pandas DataFrame) for Phase 0
- Monitor performance in development
- If performance issues arise, consider migrating to Postgres (Replit provides access)
- Defer optimization until after Phase 0 gates pass

**Owner**: Backend developer

---

## Summary

**Current State**: Flask app with Jinja templates, existing `/classify` endpoint (partial JSON support), no `/api/*` endpoints, no React.

**Phase 0 Goal**: Stabilize API contracts, add missing endpoints, add contract tests, establish pre-commit gate — **all before React install**.

**Phase 1 Goal**: Optional tooling validation.

**Phase 2 Goal**: Install React/Vite/Tailwind, build minimal vertical slice, keep app runnable.

**Key Principle**: **No frontend changes until API contracts are stable and tested.**

---

## Next Actions

1. ✅ Review this audit document
2. ⏳ Approve Phase 0 plan
3. ⏳ Begin Phase 0 implementation (Steps 0.1-0.10)
4. ⏳ Validate pre-commit gate works
5. ⏳ Proceed to Phase 2 only after Phase 0 gates pass
