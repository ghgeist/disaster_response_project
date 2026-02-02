---
created: 2026-01-29
updated: 2026-02-01
status: active
type: implementation_plan
supersedes:
  - 2026_01_29_dashboard_design_implementation_plan.md
  - 2026_01_29_figma_code_analysis.md
related:
  - 2026_01_29_storm_signal_dashboard_design_spec.md
  - 2026_01_29_storm_signal_system_spec.md
prs:
  - "86 (Phase 3: metrics endpoint)"
  - "87 (Phase 4: classification enhancement)"
---

# Storm Signal Dashboard — Implementation Plan

> **Goal**: Build the Storm Signal Dashboard React SPA with Flask API backend, integrating Figma-generated code with minimal churn and avoiding CI hell.

> **Status**: Backend Phases 0–4 complete (feed, metrics, categories, classify). React integration (Phases 5–6) pending.

> **Key Principle**: **Stabilize API contracts before frontend integration. Use simplified inline utilities.**

> **Data Reality Gate** (applies to all `/api/*` endpoints)
> - **Invariant**: All `/api/*` endpoints tolerate `None`/NaN inputs from pandas/DB and **never emit NaN or Infinity in JSON**.
> - **Gate**: Each endpoint must have at least one null-realism test (fixture with None/NaN values; assert 200 + valid shape).
> - **Boundary**: All conversions at the boundary use safe helpers (no raw `int(x)` / `float(x)` on DB/pandas values).
> - **Done means**: Endpoint passes its contract tests, its null-realism test, and a no-NaN/Infinity-in-JSON invariant test.
> - *This turns “remember to handle NaN” into “it’s part of done.”*

> **Vendor Rule**: **Treat `_vendor/figma_make/` as read-only upstream until API contracts are stable. Copy build output only, don't edit vendor source unless necessary.**

---

## Executive Summary

**Current State**:
- ✅ Flask app with Jinja templates (existing)
- ✅ `/classify` endpoint exists with partial JSON support
- ✅ Figma-generated React code in `_vendor/figma_make/` (~70% complete)
- ✅ `GET /api/feed`, `GET /api/metrics`, `GET /api/categories`, `POST /api/classify` implemented (Phases 0–4)
- ❌ No React integration yet

**Approach**: Simplified implementation using inline utilities instead of separate files. Target desktop-only design (1280px+).

**Estimated Time**: 15-19 hours total (includes Phase 0.4 API contract stubs)

---

## 1. Repo Inventory

### Backend Entrypoints & Routes

**Primary Entrypoint**: `run.py`
- Creates Flask app via factory pattern (`app.app.create_app`)
- Handles Replit deployment (reads `$PORT` env var)
- Runs on `0.0.0.0:$PORT` (default 5000)

**Route Blueprints** (`app/routes/`):
- `home_bp`: `/`, `/index`, `/favicon.ico` (Jinja templates)
- `classification_bp`: `/go`, `/classify` (supports JSON via `format=json` or `Content-Type: application/json`)
- `health_bp`: `/health`, `/health/detailed`, `/api/model-health`, `/api/performance-diagnostics`

**Existing API Endpoints**:
- `POST /classify` (or `GET /classify?query=...&format=json`) — Returns dict when `_is_json_request()` is True
  - Response shape: `{query, use_hierarchy, raw: {predictions, probabilities, labels}, fixed: {...}, violations, metrics}`
- `GET /health` — Returns `{'status': 'ok'}` (200)

**API Endpoints** (per spec):
- `GET /api/feed` — Implemented (Phase 2)
- `GET /api/metrics` — Implemented (Phase 3, PR #86)
- `GET /api/categories` — Implemented (Phase 1)

### Frontend Code (Vendored Artifact)

**Location**: `_vendor/figma_make/`

**Status**: ~70% complete (vendored from Figma export)
- ✅ Three-panel resizable layout (Feed, Metrics, Classification)
- ✅ Mock data structures matching design spec
- ✅ UI components (shadcn/ui style)
- ✅ Tailwind CSS styling
- ⚠️ Mock data that needs real database integration
- ⚠️ No backend API integration yet

**Vendor Management Rule**:
- **Treat as read-only upstream** until API contracts are stable
- **Integration strategy**: Copy build output to `app/static/dashboard/` (Phase 5) without editing vendor source
- **Exception**: Only edit vendor source if absolutely necessary (e.g., critical bug fix)
- **Rationale**: Prevents accidental refactor/ownership of vendor code, keeps upstream clean

**Components**:
- `App.tsx` — Main app with resizable panels
- `FeedPanel.tsx` — Feed list with filters
- `MetricsPanel.tsx` — Metrics cards and charts
- `ClassificationPanel.tsx` — Message classification form

**Dependencies**: Already in `package.json` (React 18, Vite, Tailwind, recharts, etc.)

### Services & Data Layer

**Services** (`app/services/`):
- `model_service.py` — Model loading and prediction
- `data_service.py` — Database access (SQLite via SQLAlchemy)
- `category_mapper.py` — Category name mapping (needs enhancement)

**Database**:
- SQLite: `data/02_stg/stg_disaster_response.db`
- Table: `stg_disaster_response`
- Columns: `id`, `message`, `original`, `genre`, plus 36 category columns
- **No timestamp column** → Generate simulated timestamps

---

## 2. Data Structure Mapping

### Figma Mock Data Structure

```typescript
interface SignalItem {
  id: string;                    // "SIG-1001"
  timestamp: Date;                 // Generated (last 6 hours)
  source: string;                  // "X", "News", "Direct Report"
  content: string;                 // Message text
  originalContent?: string;        // If translated
  language: Language;             // "en", "es", "fr", "ht"
  riskLevel: RiskLevel;            // "HIGH" | "MEDIUM" | "LOW"
  categories: string[];            // Top 3 category names
  classifications: Classification[]; // All detected categories with confidence
  isTranslated: boolean;
}
```

### Database Schema Mapping

| Database Column | Maps To | Notes |
|----------------|---------|-------|
| `id` | `SignalItem.id` → `SIG-{id}` | Format as signal ID |
| `message` | `SignalItem.content` | Truncate to 120 chars for preview |
| `original` | `SignalItem.originalContent` | Check if differs from `message` |
| `genre` | `SignalItem.source` | Map: `direct`→"Direct Report", `news`→"News", `social`→"X" |
| 36 category columns | `SignalItem.classifications` | Binary labels → probabilities from model |

**Category Name Mapping** (Internal → Display):
- Most categories: `medical_help` → `Medical Help` (simple `.replace('_', ' ').title()`)
- Special cases (~10): `search_and_rescue` → `Search & Rescue`, `infrastructure_related` → `Infrastructure`
- Use inline dict + function (no separate utility file)

---

## 3. Desktop Design Dimensions

### Viewport Size Standards

**Design Target**: Desktop-only application (no responsive breakpoints needed)

**Common Desktop Viewport Sizes (2024-2025)**:
- **1920×1080** (Full HD) - 22.63% of users
- **1366×768** (HD) - 17.42% of users  
- **1536×864** - 11.13% of users
- **1280×720** (HD) - 6.05% of users
- **1440×900** - 5.63% of users
- **1600×900** - 3.69% of users

**Minimum Supported Width**: **1280px** (covers 95%+ of desktop users)

**Recommended Design Width**: **1440px** (good balance for 3-panel layout)

**Maximum Design Width**: **1920px** (use max-width constraints)

### Panel Width Guidelines

**Current Design Spec**:
- Left Panel (Feed): Default 40%, min 300px, max 60%
- Center Panel (Metrics): Default 35%, min 300px
- Right Panel (Controls): Default 25%, min 250px, collapsible

**Best Practices**:
- Feed panel: 300px min allows ~40-50 characters per line (readable)
- Metrics panel: 300px min fits chart + cards comfortably
- Classification panel: 250px min fits form + results
- Left panel max 60% prevents excessive width (hard to scan)

**Content Density**:
- Feed items: ~120-140px height per item (comfortable scrolling)
- Metrics cards: ~80-100px height (fits 2 cards vertically)
- Charts: Minimum 150px height for readability
- Form inputs: Standard 40-44px height

**Typography**:
- Headers: 14-16px (small), 18-20px (main titles)
- Body text: 12-14px (readable at 1280px+)
- Labels/Meta: 10-11px (timestamps, badges)
- Code/Mono: 10-12px (IDs, percentages)

**Testing**: Test at 1280px, 1440px, and 1920px widths. Verify panel resizing works at each width.

---

## 4. Implementation Plan

### Phase 0: Validation & Quick Wins (1 hour)

**Objective**: Validate existing infrastructure, create API contract stubs, and capture low-hanging fruit.

#### Step 0.1: Test Existing `/classify` JSON Endpoint
- Verify `Content-Type: application/json` works
- Test `?format=json` query param
- Document current response format
- **Decision Point**: Can we reuse this or need new endpoint?

#### Step 0.2: Build React App Standalone
- Run `npm install` and `npm run build` in `_vendor/figma_make`
- Verify it runs with mock data
- Test resizable panels and UI components
- **Done means**: React app builds and runs standalone

#### Step 0.3: Create Category Display Name Mapping
- Create inline dict for ~10 special cases in `api.py`
- Use simple `.replace('_', ' ').title()` for others
- **Done means**: Mapping function ready for use

#### Step 0.4: Create API Contract Stubs
**Files**: `app/routes/api.py` (new blueprint)
- Create `api_bp` blueprint
- Implement stub endpoints that return **hard-coded JSON** with exact shapes React expects:
  - `GET /api/feed` → Return 1 hard-coded `SignalItem` matching Figma interface
  - `GET /api/metrics` → Return 1 hard-coded metrics object matching Figma `SYSTEM_METRICS`
  - `GET /api/categories` → Return 1 hard-coded category with groups
  - `POST /api/classify` → Return 1 hard-coded classification result
- **Purpose**: Detect contract drift before touching pandas/model plumbing
- **Validation**: Test React app can consume these stubs (update fetch URLs to point to Flask)
- **Done means**: All endpoints return valid JSON matching React expectations (even if hard-coded)

**Phase 0 Gate**: Existing endpoints validated, React app builds, category mapping ready, API contract stubs validate React integration works.

#### Phase 0 Completion Notes
- ✅ `/classify?format=json` returns JSON; JSON POSTs to `/classify` are blocked by CSRF (use `/api/classify` for React).
- ✅ React app builds in `_vendor/figma_make` (`npm install` + `npm run build`); Vite chunk warning noted.
- ✅ Category display mapping and groups defined inline in `app/routes/api.py`.
- ✅ API contract stubs implemented in `app/routes/api.py`, registered in `app/routes/__init__.py`, and CSRF-exempt via `app/extensions.py`.
- ✅ Lightweight contract tests added: `tests/test_api_contract_stubs.py` (4 tests), passing via `python scripts/run_tests.py tests/test_api_contract_stubs.py -q`.
- ⚠️ Do not commit `_vendor/figma_make/node_modules/` or `_vendor/figma_make/dist/` (build artifacts).

---

### Phase 1: Backend API Foundation (1.5-2 hours)

**Objective**: Replace API contract stubs with real database/model integration.

**Note**: API blueprint already created in Phase 0.4. This phase swaps hard-coded responses for real data.

#### Step 1.2: Implement Severity Calculation (Inline)
**Files**: `app/routes/api.py`
- Add inline function:
  ```python
  def calculate_severity(probabilities: dict) -> str:
      CRITICAL_CATEGORIES = {'medical_help', 'medical_products', 'search_and_rescue',
                            'water', 'food', 'shelter', 'security', 'hospitals'}
      critical_count = sum(1 for cat, prob in probabilities.items() 
                          if cat in CRITICAL_CATEGORIES and prob > 0.5)
      max_conf = max(probabilities.values()) if probabilities else 0.0
      if critical_count >= 2 or max_conf > 0.85: return 'HIGH'
      elif critical_count >= 1 or max_conf > 0.70: return 'MEDIUM'
      return 'LOW'
  ```
- **No separate utility file** — keep inline

#### Step 1.3: Implement Category Display Name Mapping (Inline)
**Files**: `app/routes/api.py`
- Add inline dict + function:
  ```python
  CATEGORY_DISPLAY_NAMES = {
      'search_and_rescue': 'Search & Rescue',
      'infrastructure_related': 'Infrastructure',
      'aid_centers': 'Aid Centers',
      # ... ~10 special cases
  }
  
  def to_display_name(internal: str) -> str:
      return CATEGORY_DISPLAY_NAMES.get(internal, internal.replace('_', ' ').title())
  ```

#### Step 1.4: Implement `GET /api/categories` Endpoint
**Files**: `app/routes/api.py`
- Load category metadata from `DataService.get_data()`
- Calculate category counts: `df[category_cols].sum().to_dict()` (5 lines)
- Return category groups matching design spec
- Use inline display name mapping

**Done means**: Endpoint exists, returns category metadata with counts.

#### Phase 1 Completion Notes
- ✅ Added inline severity calculation helper in `app/routes/api.py`.
- ✅ `/api/categories` now returns real category counts from `DataService` plus groups.
- ✅ Contract stub tests still pass: `python scripts/run_tests.py tests/test_api_contract_stubs.py -q`.
- ⚠️ **pytest.ini fix**: Updated `norecursedirs` to exclude `.venv`, `venv`, `env`, `.git`, `__pycache__`, `.pytest_cache`, `build`, `dist`, `node_modules` (previously only excluded `scripts`). This was causing pytest to recurse into directories it shouldn't, requiring ~30 minutes to diagnose and fix.
- 🧭 **Prevent repeat**: Keep `norecursedirs` updated when new tool folders appear (e.g., virtual envs, build outputs, node artifacts). If pytest suddenly slows or hangs, check recursion paths first before debugging tests.

---

### Phase 2: Feed Endpoint (3-4 hours)

**Phase 2 Gate (merge criteria)**:
- ✅ Ruff passes (at minimum catching unused vars/imports + dead stores)
- ✅ Endpoint tests cover:
  - empty result set (including pagination invariants)
  - unexpected enum value(s)
  - null/NaN conversion safety
- ✅ If randomness is used: tests assert invariants/ranges, not exact values
- ✅ API response shape matches the documented contract in Section 6

**Objective**: Create feed endpoint with simulated timestamps and source mapping.

#### Step 2.1: Implement Timestamp Generation (Inline)
**Files**: `app/routes/api.py`
- Add inline function:
  ```python
  from datetime import datetime, timedelta
  import random
  
  def generate_timestamp(index: int, total: int) -> datetime:
      # Spread over last 6 hours, most recent first
      hours_ago = (index / total) * 6
      return datetime.now() - timedelta(hours=hours_ago)
  ```

#### Step 2.2: Implement Source Mapping (Inline)
**Files**: `app/routes/api.py`
- Add inline dict:
  ```python
  GENRE_TO_SOURCE = {
      'direct': 'Direct Report',
      'news': 'News',
      'social': 'X'
  }
  ```

#### Step 2.3: Implement `GET /api/feed` Endpoint
**Files**: `app/routes/api.py`
- Query params: `limit` (default 25), `offset` (default 0), `categories[]` (filter)
- Load messages from `DataService.get_data()`
- **Performance Guardrail**: Use binary labels + simulated confidences initially
  - Read binary category labels from database (0/1)
  - Convert to probabilities: label=0 → `0.1 + random(0, 0.1)`, label=1 → `0.85 + random(0, 0.15)` (simulated)
  - **Alternative**: Run model only for top N items (e.g., 10) during first integration
  - **Rationale**: Per-item inference on 25+ rows can be slow. Demo system behavior without full compute cost.
  - **Future**: Decide whether real inference belongs in feed (may need caching/pre-computation)
- Calculate severity for each message (reuse from Phase 1)
- Generate simulated timestamps (inline function)
- Map categories to display names (reuse from Phase 1)
- Map genre to source (inline dict)
- Implement pagination
- Return JSON array matching Figma `SignalItem` structure
- **Null-realism (Data Reality Gate)**:
  - Add null-realism fixture: at least one row with `genre=None`, one category column NaN, and one optional text field None
  - Add test: `test_feed_handles_nulls_and_nans()` asserting 200 + valid response shape + pagination contract

**Done means**: Endpoint passes Phase 2 Quality Gates; Tripwires #1–#3 and #6 are covered by tests; Data Reality Gate (null-realism test + no NaN/Infinity in JSON). Endpoint exists, returns paginated feed items with all required fields. Uses binary labels + simulated confidences (or limited model inference) to avoid performance issues.

#### Phase 2 Completion Notes
- ✅ Timestamp generation: `generate_timestamp_for_id(raw_id)` in `app/routes/api.py` spreads over last 6 hours, deterministic per id.
- ✅ Source mapping: `genre_to_source(genre)` with `GENRE_TO_SOURCE` dict; `social` maps to `X` (default source).
- ✅ `GET /api/feed`: Real data from `DataService.get_data()`, query params `limit` (default 25, max 100), `offset` (default 0), `categories[]` (filter by internal names). Binary labels + simulated confidences (label=0 → `0.1 + random(0,0.1)`, label=1 → `0.85 + random(0,0.15)`), severity from `calculate_severity`, pagination with `page`, `limit`, `total`, `totalPages`. Response shape matches Figma `SignalItem`.
- ✅ Contract tests still pass: `python scripts/run_tests.py tests/test_api_contract_stubs.py -q` (or `.venv\Scripts\python -m pytest tests/test_api_contract_stubs.py -q`).
- ✅ Regression tests added to avoid feed/pagination drift: offset in-range, out-of-range clamp, empty dataset, limit bounds (0→1, 999→100), filter+offset clamp, genre/NaN/unknown→X, classification inclusion (conf > 0.5). Stub `StubDataService` used so pagination tests are deterministic; does not block Phase 5 (React fetch) or future steps.

---

### Phase 3: Metrics Endpoint (1.5-2 hours)

**Objective**: Create metrics endpoint with real category counts and simulated volume data.

#### Step 3.1: Implement `GET /api/metrics` Endpoint
**Files**: `app/routes/api.py`
- `vol_today`: Simulated count (can use database size as base)
- `flagged_pct`: Simulated percentage
- `top_categories`: **Real** counts from database (top 5-7)
  - Use pandas: `df[category_cols].sum().sort_values(ascending=False).head(7)`
- `trend_data`: Simulated time series (6 hours, hourly intervals)
  - Can be static array or simple random walk
- Return JSON matching Figma `SYSTEM_METRICS` structure
- **Null-realism (Data Reality Gate)**:
  - Add null-realism fixture: at least one category column NaN
  - Add test: `test_metrics_handles_nulls_and_nans()` asserting 200 + numeric fields present + no NaN in JSON output

**Done means**: Endpoint passes Phase 2 Quality Gates; NaN safety and empty-data behavior are explicitly tested; Data Reality Gate (null-realism test + no NaN/Infinity in JSON). Endpoint exists, returns metrics with real category counts and simulated volume/trends.

#### Phase 4 Completion Notes (PR #87)
- ✅ **PR #87** (feature/phase-4-classification-enhancement) merged: `POST /api/classify` replaced stub with real implementation.
- ✅ `classify()` in `app/routes/api.py`: parses JSON `message`, validates input, calls `process_prediction_result(model_service, message)`; builds simplified response via `_build_simplified_classification()`.
- ✅ Severity: `calculate_severity(probabilities)` (Phase 1). Category volume: real counts from `DataService.get_data()` (Phase 3). Per-label thresholds: `model_service.get_thresholds_map()` for inclusion.
- ✅ Response shape: `categories` (name, confidence, volume), `severity`, `maxConfidence`, `avgConfidence`. Categories included where probability ≥ threshold; sorted by confidence, capped at 10.
- ✅ Validation: empty/missing message → 400; missing model_service → 503; ModelServiceError/ValueError → 503.
- ✅ **NaN/None safety (Tripwire #3)**: All probability and threshold values pass through `_safe_float_prob()` before comparison, max/avg, or JSON; `critical_count` and `critical_probabilities` in `calculate_severity` use it; above-threshold filter and sort use it; category `confidence`, `maxConfidence`, and `avgConfidence` use it. Category keys use `_safe_category_display()` so None/NaN keys do not crash. Thresholds from `thresholds_map` are sanitized with `_safe_float_prob()` before comparison. Data-service failure is isolated so classification still returns 200 with empty volumes. `GET /api/categories` uses `fillna(0).sum()` and `_safe_label_value(count)` for NaN-safe counts.
- ✅ Tests: `test_api_classify_contract` (shape), `test_api_classify_empty_message_returns_400`, `test_api_classify_no_model_service_returns_503`. Contract tests: `python scripts/run_tests.py tests/test_api_contract_stubs.py -q` (20 passed). Ruff passes on `app/routes/api.py`.

---

#### Phase 3 Completion Notes (PR #86)
- ✅ **PR #86** (feature/phase-3-metrics-endpoint) merged: `GET /api/metrics` returns real data from `DataService`.
- ✅ `_build_metrics_response(df, category_columns)` in `app/routes/api.py`: builds `volToday`, `flaggedRate`, `topCategories`, `trendData`; uses `df[cats].fillna(0)` for NaN-safe category sums; top 7 categories via `sums.sort_values(ascending=False).head(7)`; counts coerced with `_safe_label_value(count)`.
- ✅ Simulated volume/trends: `vol_today = n * 100` (0 when empty); `flagged_pct` from `2.0 + (flagged/n)*5.0` capped at 10.0; `trendData` is 7 entries (`TREND_LABELS`) with static counts when `n > 0`, else zeros.
- ✅ Empty dataset: `volToday=0`, `flaggedRate=0.0`, `topCategories=[]`, `trendData` has 7 entries with `count=0`.
- ✅ Refactor: removed `_safe_count` and `_build_stub_metrics`; shared `_safe_label_value` used in feed and metrics for consistent NaN/None handling.
- ✅ Tests: `test_api_metrics_contract` (shape), `test_metrics_empty_dataset` (empty result set), `test_metrics_nan_in_categories_does_not_crash` (NaN safety). Contract tests still pass: `python scripts/run_tests.py tests/test_api_contract_stubs.py -q`.

---

### Phase 4: Classification Enhancement (1.5-2 hours)

**Objective**: Enhance existing `/classify` endpoint or create `/api/classify` wrapper.

#### Step 4.1: Decision Point
- **Option A**: Add `?simple=true` mode to existing `/classify`
  - Transform existing response to simplified format
  - Add severity calculation
  - Add category volume context
- **Option B**: Create `/api/classify` wrapper endpoint
  - Thin wrapper around existing logic
  - Returns simplified format

**Recommendation**: Test Phase 0 first, then decide. Option A is less code if response transformation is simple.

#### Step 4.2: Implement Chosen Approach
**Files**: `app/routes/api.py` or `app/routes/classification.py`
- Add severity calculation (reuse from Phase 1)
- Add category volume context (reuse from Phase 3)
- Return simplified JSON matching React component expectations

**Done means**: Classification endpoint returns simplified JSON with severity and volume context; Data Reality Gate (null-realism test + no NaN/Infinity in JSON).

---

### Phase 5: Frontend Integration (4-5 hours)

**Objective**: Integrate React app with Flask backend.

#### Step 5.1: Copy React Build Output (Vendor Rule)
- Build React app: `cd _vendor/figma_make && npm run build`
- Copy `dist/` contents to `app/static/dashboard/`
- **Vendor Rule**: Don't edit `_vendor/figma_make/` source files
- **If changes needed**: Make minimal edits only if absolutely necessary, document why

#### Step 5.2: Create Flask Route for React App
**Files**: `app/routes/api.py` or new `app/routes/dashboard.py`
- Add route: `@api_bp.route('/dashboard')` or `@api_bp.route('/dashboard/<path:path>')`
- Serve `index.html` for SPA routing
- Ensure static assets are accessible

#### Step 5.3: Update React Components
**Files**: `_vendor/figma_make/src/app/components/dashboard/*.tsx`
- Replace `MOCK_SIGNALS` with `fetch('/api/feed')`
- Replace mock classification with `POST /api/classify` (or `/classify?simple=true`)
- Replace mock metrics with `fetch('/api/metrics')`
- Update category mapping to use API response
- Add error handling and loading states

#### Step 5.4: Test End-to-End
- Verify React app loads
- Test feed displays real data
- Test classification works
- Test metrics display
- Verify panel resizing works

**Done means**: React app loads, displays real data, all features work end-to-end.

#### Phase 5 Completion Notes
- ✅ **Branch**: `feature/phase-5-frontend-integration`.
- ✅ **Step 5.1**: Vite `base: '/static/dashboard/'` in `_vendor/figma_make/vite.config.ts`; React app built with `npm run build`; `dist/` contents copied to `app/static/dashboard/` (index.html, assets/*.js, assets/*.css).
- ✅ **Step 5.2**: Flask routes added in `app/routes/api.py`: `GET /api/dashboard` and `GET /api/dashboard/<path:path>` serve `app/static/dashboard/index.html` for SPA routing; static assets served at `/static/dashboard/assets/`.
- ✅ **Step 5.3**: React components updated (minimal vendor edits): `App.tsx` fetches `/api/feed?limit=50`, maps items (timestamp string → Date), passes loading/error to `FeedPanel`; `FeedPanel` accepts optional `loading`/`error` and shows loading/error states; `MetricsPanel` fetches `/api/metrics`, maps `volToday`/`trendData` to `volumeToday`/`flaggedHistory`, shows LIVE badge; `ClassificationPanel` POSTs to `/api/classify` with `{ message }`, maps response `categories` (name, confidence, volume) to result shape, shows classify error when applicable.
- ✅ **Step 5.4**: Manual check: `GET /api/dashboard` returns 200 and index.html; feed and metrics APIs return JSON; dashboard loads and fetches real data from Flask.

---

### Phase 6: Polish & Testing (2-3 hours)

**Objective**: Add polish, error handling, and testing.

#### Step 6.1: Add Pagination UI
- Add "Load More" button to feed panel
- Implement infinite scroll (optional)

#### Step 6.2: Add Error Boundaries
- Add React error boundaries
- Add error states in components
- Display user-friendly error messages

#### Step 6.3: Add Loading States
- Add loading skeletons (some already exist)
- Add spinners for async operations

#### Step 6.4: Test at Key Viewport Sizes
- Test at 1280px width (minimum)
- Test at 1440px width (recommended)
- Test at 1920px width (maximum)
- Verify panel resizing at each width

#### Step 6.5: Cross-Browser Testing
- Chrome/Edge (Chromium) - primary
- Firefox - secondary
- Safari - if Mac users expected

**Done means**: App is polished, handles errors gracefully, works across viewports and browsers.

---

## 5. File Structure

### New Files to Create

```
app/
  routes/
    api.py                    # NEW: API blueprint with inline utilities
  static/
    dashboard/                # NEW: React build output
      index.html
      assets/
        *.js
        *.css
```

### Files to Modify

```
app/
  routes/
    classification.py         # MODIFY: Add ?simple=true mode (optional)
  app.py                      # MODIFY: Register API blueprint
```

### Files NOT Needed (Simplified Approach)

~~`app/utils/severity.py`~~ → Inline function in `api.py`  
~~`app/utils/category_display.py`~~ → Inline dict + function in `api.py`  
~~`app/utils/timestamp_generator.py`~~ → Inline datetime math  
~~`app/utils/source_mapper.py`~~ → Inline dict lookup

**Rationale**: These are simple operations (5-10 lines each) that don't need separate modules. Keep code close to where it's used for easier maintenance.

**Boundary conversion rule**: When reading from pandas/DB, never call `int(x)` / `float(x)` directly. Use a tiny safe conversion helper (inline or local to `api.py`) for any external value. That’s enough for an AI agent to follow reliably.

---

## 6. API Response Formats

### `GET /api/feed`

**Query Params**:
- `limit` (int, default: 25)
- `offset` (int, default: 0)
- `categories[]` (array, optional): Filter by category names

**Response**:
```json
{
  "items": [
    {
      "id": "SIG-1001",
      "timestamp": "2026-01-29T15:24:59Z",
      "source": "X",
      "content": "Urgent: Water rising rapidly...",
      "originalContent": null,
      "language": "en",
      "riskLevel": "HIGH",
      "categories": ["Water", "Search & Rescue", "Floods"],
      "classifications": [
        {"category": "Water", "confidence": 0.92},
        {"category": "Search & Rescue", "confidence": 0.88}
      ],
      "isTranslated": false
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 25,
    "total": 150,
    "totalPages": 6
  }
}
```

### `GET /api/metrics`

**Response**:
```json
{
  "volToday": 14502,
  "flaggedRate": 4.2,
  "topCategories": [
    {"name": "Medical Help", "count": 1247},
    {"name": "Water", "count": 892},
    {"name": "Food", "count": 756}
  ],
  "trendData": [
    {"time": "6h ago", "count": 45},
    {"time": "5h ago", "count": 120},
    {"time": "4h ago", "count": 80},
    {"time": "3h ago", "count": 210},
    {"time": "2h ago", "count": 150},
    {"time": "1h ago", "count": 95},
    {"time": "Now", "count": 60}
  ]
}
```

### `GET /api/categories`

**Response**:
```json
{
  "categories": [
    {"internal": "medical_help", "display": "Medical Help", "count": 1247},
    {"internal": "water", "display": "Water", "count": 892}
  ],
  "groups": {
    "Critical Needs": ["Medical Help", "Medical Products", "Search & Rescue", "Water", "Food", "Shelter", "Security", "Hospitals"],
    "Infrastructure": ["Transport", "Buildings", "Electricity", "Tools", "Shops", "Aid Centers", "Other Infrastructure"],
    "Weather": ["Floods", "Storm", "Fire", "Earthquake", "Cold", "Other Weather"],
    "Other": ["Missing People", "Refugees", "Death", "Clothing", "Money", "Other Aid", "Military", "Child Alone", "Request", "Offer", "Direct Report"]
  }
}
```

### `POST /api/classify` (or `/classify?simple=true`)

**Request**:
```json
{
  "message": "Urgent: Water rising rapidly..."
}
```

**Response**:
```json
{
  "categories": [
    {"name": "Water", "confidence": 0.92, "volume": 892},
    {"name": "Search & Rescue", "confidence": 0.88, "volume": 421}
  ],
  "severity": "HIGH",
  "maxConfidence": 0.92,
  "avgConfidence": 0.90
}
```

---

## 7. Risk Register

### Risk 1: Performance Issues with Model Predictions

**Impact**: High — Running model on every feed item (25+ rows) can be slow

**Mitigation**:
- **Phase 0.4**: API contract stubs validate React integration before model plumbing
- **Phase 2.3**: Use binary labels + simulated confidences initially (no model inference)
  - Alternative: Run model only for top N items (e.g., 10) during first integration
  - Demo system behavior without full compute cost
- **Future decision**: Determine if real inference belongs in feed (may need caching/pre-computation)
- Start without caching (simpler), add if performance issues arise

### Risk 2: API Contract Drift

**Impact**: High — Frontend breaks if API shape changes

**Mitigation**:
- **Phase 0.4**: API contract stubs return hard-coded JSON matching React expectations
  - Fastest way to detect contract drift before touching pandas/model plumbing
  - Validates React can consume API responses before real implementation
- Keep API responses simple and well-documented
- Test API endpoints manually during development
- Consider JSON Schema validation (future enhancement)

### Risk 3: Frontend Integration Breaks Existing Routes

**Impact**: Medium — App becomes unusable during migration

**Mitigation**:
- Keep existing Jinja routes (`/`, `/go`) working
- Serve React app at `/dashboard` (separate route)
- Gradual migration approach

### Risk 4: Viewport Compatibility

**Impact**: Low — App may not work well at certain widths

**Mitigation**:
- Test at key viewport sizes (1280px, 1440px, 1920px)
- Use resizable panels (already implemented)
- Mobile warning banner (already implemented)

---

## 8. Testing Strategy

### Manual Testing (Primary)

**Phase 0-4**: Manual API testing
- Test endpoints with `curl` or Postman
- Verify JSON responses match expected format
- Test edge cases (empty results, invalid params)

**Phase 5-6**: Manual UI testing
- Test React app in browser
- Test all features end-to-end
- Test at different viewport sizes

### Unit Tests (Optional)

**If Time Permits**:
- Severity calculation function
- Category name mapping function
- Timestamp generation function

**Framework**: pytest (existing)

### Integration Tests (Future)

**If Needed**:
- API endpoint contract tests (feed pagination, genre/NaN, classification threshold covered in `tests/test_api_contract_stubs.py`)
- React component tests
- E2E tests

---

## 9. Next Steps

### Immediate Actions (Today)
1. ✅ Review this implementation plan
2. ✅ **Phase 0.1**: Test existing `/classify` JSON endpoint (15 min)
3. ✅ **Phase 0.2**: Build React app standalone and verify (15 min)
4. ✅ **Phase 0.3**: Create category display name mapping (10 min)
5. ✅ **Phase 0.4**: Create API contract stubs with hard-coded JSON (30 min)
   - Validate React can consume stubs before real implementation
6. ✅ Start Phase 1: Backend API Foundation (swap stubs for real data)

### Short-term (This Week)
1. ✅ Complete Phases 1-4 (Backend API) - **8-11 hours**
2. ✅ Complete Phase 5 (Frontend Integration) - **4-5 hours**
3. ✅ Basic testing at key viewport sizes

### Medium-term (Next Week)
1. ✅ Phase 6 (Polish & Testing) - **2-3 hours**
2. ✅ Cross-browser testing
3. ✅ Documentation updates

**Total Estimated Time**: 14-18 hours

---

## 10. Key Decisions Made

1. **Vendor Management**: Treat `_vendor/figma_make/` as read-only upstream until API contracts stable
2. **API Contract Stubs**: Return hard-coded JSON first (Phase 0.4) to detect contract drift early
3. **Performance Guardrail**: Use binary labels + simulated confidences in feed (avoid per-item model inference)
4. **Simplified Utilities**: Use inline functions instead of separate files for simple operations
5. **Severity Calculation**: Use probabilities from model (not binary labels)
6. **Timestamp Generation**: Generate simulated timestamps (no DB column exists)
7. **Category Mapping**: Inline dict + function (no separate utility)
8. **Desktop-Only Design**: Target 1280px+ width, no responsive breakpoints
9. **Classification Endpoint**: Test Phase 0 first, then decide on enhancement approach
10. **No Contract Tests Initially**: Manual testing first, add automated tests if needed

---

## 11. CI & Correctness Tripwires (Phase 2+)

**Purpose**: Encode the recurring failure modes we've already paid for, so future PRs (and AI agents) don't relearn them.

**Tripwires Table (living)**:

| # | Tripwire | Severity | Invariant violated | Invariant | Primary Catch |
|---|----------|----------|--------------------|-----------|---------------|
| 1 | Unknown genres → random social sources | Low | Unknown enums not explicitly handled | Unknown enums must be explicitly handled (warn+default OR 4xx), never ambiguous dict.get() fallback | Unit test + sentinel default pattern |
| 2 | Pagination returns page 0 for empty results | Medium | Pagination not 1-indexed when empty | Pagination is always 1-indexed, even when empty | Boundary tests |
| 3 | NaN values crash endpoint | Medium | Null/NaN in conversion path | Null/NaN must never crash conversion logic | Safe conversion helper + fixture with NaN |
| 4 | Unused variable (slice_size) | Low | Dead store in committed code | No dead stores in committed code | Ruff F841 |
| 5 | Unused function (_build_stub_feed_item) | Medium | Dead code path after refactor | No dead code paths left behind after refactor | Ruff / coverage smell |
| 6 | Probability formula makes label=0 exceed threshold | High | Simulated probs cross decision boundary | Simulated probs must respect threshold semantics (no overlap across decision boundary) | Range test / property test |
| 7 | Duplicate exception handlers | Low | Handlers don't differ meaningfully | Exception handlers either differ meaningfully or are consolidated | Pylint / review checklist |

**Prevention by category**:
- **Linter-catchable**: Enforce Ruff (F841/F401 minimum) + optional Pylint "duplicate-except"
- **Test-catchable**: Add edge-case tests: unknown inputs, empty results, NaN handling, probability-range invariants
- **Data edge cases**: Tests must include None and NaN values in fixtures for any endpoint touching pandas/DB
- **Design-review**: Write down API edge contracts + compute formula ranges before coding

This keeps the "guide" in the repo without turning it into ceremony.

---

**Document Status**: ✅ Complete - Ready for implementation

**Last Updated**: 2026-02-01
