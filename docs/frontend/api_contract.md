# React Dashboard API Contract

The React dashboard consumes Flask endpoints under `/api/*` using relative URLs. The API is designed to be low-latency and resilient; the UI displays fallback messaging when a request fails.

## Error Handling Patterns
- **HTTP errors**: React treats non-200 responses as failures and surfaces a panel-level error state.
- **Empty payloads**: UI falls back to placeholder data to avoid blank screens.
- **Network failures**: Errors are displayed in-panel without blocking the rest of the dashboard.

---

## `GET /api/feed`
**Purpose:** Fetch the real-time message feed.

**Query Params**
- `limit` (int): max number of items (default 25, capped at 100).
- `offset` (int): pagination offset.
- `categories[]` (string[]): category filters.

**Response Format**
```json
{
  "items": [
    {
      "id": "string",
      "timestamp": "2024-01-01T12:00:00Z",
      "source": "Direct Report",
      "content": "Message text",
      "riskLevel": "LOW",
      "categories": ["Medical Help", "Water"]
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 25,
    "total": 250,
    "totalPages": 10
  }
}
```

**Example**
```ts
const params = new URLSearchParams({ limit: '15' });
params.append('categories[]', 'medical_help');
const response = await fetch(`/api/feed?${params.toString()}`);
```

---

## `POST /api/classify`
**Purpose:** Classify a user-submitted message.

**Request Body**
```json
{ "message": "Need water and shelter" }
```

**Response Format**
```json
{
  "severity": "MEDIUM",
  "categories": [
    { "name": "Water", "conf": 0.86 },
    { "name": "Shelter", "conf": 0.74 }
  ]
}
```

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

**Response Format**
```json
{
  "version": "2.4.0",
  "f1_score": 0.938,
  "status": "operational",
  "hierarchy_violations": 0.0
}
```

**Example**
```ts
const response = await fetch('/api/model-info');
const payload = await response.json();
```

---

## `GET /api/model-info/dashboard`
**Purpose:** Fetch the full payload for the model information dashboard.

**Response Format**
```json
{
  "model": { "id": "MODEL_ID", "version": "2.4.0" },
  "metrics": { "f1": 0.93, "precision": 0.94, "recall": 0.92 },
  "categories": [
    { "label": "Medical Help", "f1": 0.52, "support": 432 }
  ],
  "criticalThresholds": [
    { "category": "medical_help", "threshold": 0.4 }
  ],
  "registry": [
    { "name": "MODEL_INFO.json", "size": 1234, "type": "json" }
  ]
}
```

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
