# React Dashboard API Contract

The React dashboard consumes Flask endpoints under `/api/*` using relative URLs. The API is designed to be low-latency and resilient; the UI displays fallback messaging when a request fails.

## Error Handling Patterns
- **HTTP errors**: React treats non-200 responses as failures and surfaces a panel-level error state.
- **Empty payloads**: UI falls back to placeholder data to avoid blank screens.
- **Network failures**: Errors are displayed in-panel without blocking the rest of the dashboard.

---

## `GET /api/feed`
**Purpose:** Serve the Storm Signal message feed from the committed demo-feed cache.

Historical disaster-response messages are classified once with the active production model (hierarchy-corrected positives) and stored in `app/data/demo_feed.json`. The endpoint replays that cache as a demo operational stream: classifications and category labels are real production decisions; synthetic `timestamp` values and aggregate `/api/metrics` volume/trends remain presentation simulation.

**Query Params**
- `limit` (int): max number of items (default 25, capped at 100).
- `offset` (int): pagination offset.
- `categories[]` (string[]): filter to items whose hierarchy-corrected `fixed.labels` mark the given **internal** names as positive (`value == 1`). Staging ground-truth columns are never used for display or filter decisions.

**Response Format**
```json
{
  "items": [
    {
      "id": "SIG-2",
      "timestamp": "2024-01-01T12:00:00Z",
      "source": "Direct Report",
      "content": "Message text",
      "originalContent": null,
      "language": "en",
      "riskLevel": "LOW",
      "categories": ["Weather Related", "Cold", "Other Weather"],
      "classifications": [
        {"category": "Weather Related", "confidence": 0.45},
        {"category": "Cold", "confidence": 0.26}
      ],
      "isTranslated": false
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 25,
    "total": 50,
    "totalPages": 2
  }
}
```

`timestamp` is **required** and injected by the server via a deterministic synthetic clock (`generate_timestamp_for_id`); it is not stored in the cache. Cached `raw` / `fixed` / `message_id` fields are omitted from the public JSON.

**Fail-closed behavior:** If the cache is missing/invalid, `schema_version` is unsupported, provenance hashes (`model_sha256`, `thresholds_sha256`, `labels_sha256`) do not match `ModelService.get_production_artifacts()`, or the model service is unavailable, the endpoint returns HTTP **503** with `{"error": "Feed unavailable right now."}` — there is no simulated-confidence fallback.

**Regeneration:** After promoting a new production model, rebuild the cache with `python scripts/07_operations/build_demo_feed.py` (do **not** pass `--init-ids` unless intentionally re-pinning message IDs).

**Example**
```ts
const params = new URLSearchParams({ limit: '15' });
params.append('categories[]', 'medical_help');
const response = await fetch(`/api/feed?${params.toString()}`);
```

---

## `POST /api/classify`
**Purpose:** Classify a user-submitted message.

Classification results are **hierarchy-corrected** before the simplified response is built: parent/child consistency from `apply_hierarchy()` (shared taxonomy config) is applied using the deployed per-label threshold map. The React dashboard consumes the simplified fields only.

**Request Body**
```json
{ "message": "Need water and shelter" }
```

**Response Format**
```json
{
  "severity": "MEDIUM",
  "categories": [
    {
      "name": "Water",
      "confidence": 0.86,
      "volume": 892,
      "threshold": 0.302,
      "meetsThreshold": true
    }
  ],
  "maxConfidence": 0.86,
  "avgConfidence": 0.80
}
```

**Optional debug** (`?debug=1` / `true` / `yes` / `on`): adds a `debug` object with `thresholds` plus nested `raw` and `fixed` maps (`probabilities`, `labels`) so callers can compare model output to hierarchy-adjusted decisions. This replaces the older flat `debug.probabilities` / `debug.labels` shape. Not used by the default dashboard UI.

`meetsThreshold` is `true` for categories included as positive decisions after hierarchy (including parents forced on by an active child), even when adjusted confidence is still below that label’s numeric threshold.

**Example**
```ts
await fetch('/api/classify', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: 'Need water and shelter' }),
});
```

---

## `GET /api/model-info`
**Purpose:** Fetch model metadata for the dashboard header.

Metadata is returned **only** from a provenance-valid active production bundle
(active `disaster_*_prod_*.pkl` plus stem-bound thresholds/labels whose SHA-256
values match `MODEL_INFO.json`). The endpoint uses the same
`resolve_production_artifacts(...)` contract as production inference and
`GET /api/model-info/dashboard`.

When there is no active production pickle, or when provenance validation fails,
the endpoint returns HTTP 200 with `status: "unavailable"` and does **not**
surface orphan/stale `MODEL_INFO.json` fields. Clients never receive filesystem
paths or raw exception text.

**Response Format (available)**
```json
{
  "version": "v26-09-21",
  "f1_score": 0.8975,
  "status": "production",
  "hierarchy_violations": 0.0
}
```

`f1_score` comes from `performance.optimized_f1_weighted` (else
`validation_results.optimized_f1_weighted`). It does **not** read naked
`f1_weighted` or `baseline_f1_micro`. When the explicit field is absent,
`f1_score` is `null`.

**Response Format (unavailable)**
```json
{
  "version": "unknown",
  "f1_score": null,
  "status": "unavailable",
  "hierarchy_violations": 0.0,
  "provenanceError": "Production model provenance unavailable",
  "provenanceCode": "active_model_missing"
}
```

`provenanceCode` is `active_model_missing` or `provenance_failed`.

**Example**
```ts
const response = await fetch('/api/model-info');
const payload = await response.json();
```

---

## `GET /api/model-info/dashboard`
**Purpose:** Fetch the full payload for the model information dashboard.

**Operating-point sources (one resolved production bundle):**
- Headline `metrics.f1` / `evalCriticalRecall` from promoted `MODEL_INFO.json` OP fields.
- Per-category rows and aggregate positive-class P/R from the hashed stem thresholds artifact `category_stats` (frozen-eval @ deployed thresholds).
- Displayed `criticalThresholds[].threshold` values from the validated inference map `production_artifacts.thresholds`, not from `stat["threshold"]` alone.
- Critical membership comes from `category_stats[*].type == "critical"`.
- If a critical `category_stats` key is absent from the inference map, that entry is skipped (not published as threshold `0.0`).

**Three KPI aggregation families (not one conventional P/R/F1 tuple):**
1. **Optimized Weighted F1** — mean across labels of each binary `classification_report()["weighted avg"]["f1-score"]` (includes negative + positive classes).
2. **Positive-class Weighted Precision / Recall** — support-weighted means of positive-class `category_stats` precision/recall.
3. **Critical-label mean recall** (`evalCriticalRecall`) — unweighted mean recall across critical labels only (UI secondary under recall; not a fourth interchangeable card).

**Null semantics:** When `category_stats` is missing/empty, or the payload is unavailable, `metrics.precision` and `metrics.recall` are `null` (UI: `—`). When the payload is unavailable, or `optimized_f1_weighted` is missing/invalid, `metrics.f1` is also `null` (UI: `—`). Never publish `0.0` for “not measured” headline F1 or P/R. Unavailable payloads also return `categories: []` and `criticalThresholds: []`.

**Storm Signal / Model Information headers:** While model status is loading → neutral `Loading model…`. `model.status === "production"` → `SYSTEM: OPERATIONAL`. Otherwise → `MODEL: UNAVAILABLE`. Pages that do not supply model status omit the chip. Tooltip hierarchy wording `Classifier hierarchy correction: enabled` appears only when `status === "production"`.

**Response Format**
```json
{
  "model": { "id": "MODEL_ID", "version": "v26-09-21", "status": "production", "algorithm": "lr", "algorithmName": "LogisticRegression" },
  "metrics": {
    "f1": 0.8975,
    "precision": 0.7101,
    "recall": 0.6074,
    "evalCriticalRecall": 0.6149
  },
  "categories": [
    { "key": "medical_help", "label": "Medical Help", "f1": 0.4923, "precision": 0.4032, "recall": 0.6319, "support": 432 }
  ],
  "criticalThresholds": [
    { "key": "medical_help", "label": "Medical Help", "threshold": 0.1239 }
  ],
  "registry": [
    { "name": "MODEL_INFO.json", "size": 1234, "type": "json" }
  ]
}
```

`metrics.f1` comes from `performance.optimized_f1_weighted` (else
`validation_results.optimized_f1_weighted`). It does **not** read naked
`f1_weighted` or `baseline_f1_micro`. When the explicit field is absent or
invalid, `metrics.f1` is `null` (same absence semantics as P/R).

`evalCriticalRecall` is the frozen-eval critical-label **mean** recall from promoted `MODEL_INFO.json` (`performance.eval_critical_recall`, else `validation_results`). It is a finite probability in `[0, 1]`, or `null` when missing/invalid. It is distinct from aggregate positive-class `recall`.

**Example**
```ts
const response = await fetch('/api/model-info/dashboard');
const payload = await response.json();
```

---

## `GET /api/categories`
**Purpose:** Fetch category group metadata for filters.

**Response Format**
```json
{
  "categories": [
    { "internal": "medical_help", "display": "Medical Help", "count": 432 }
  ],
  "groups": {
    "Critical Needs": ["Medical Help", "Water"]
  }
}
```

**Example**
```ts
const response = await fetch('/api/categories');
const payload = await response.json();
```

---

## `GET /api/metrics`
**Purpose:** Fetch dashboard metrics and trends.

**Response Format**
```json
{
  "summary": {
    "totalMessages": 1200,
    "criticalSignals": 85
  },
  "trends": {
    "daily": [
      { "timestamp": "2024-01-01", "count": 140 }
    ]
  }
}
```

**Example**
```ts
const response = await fetch('/api/metrics');
const payload = await response.json();
```
