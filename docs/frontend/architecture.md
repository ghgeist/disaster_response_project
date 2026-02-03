# React Dashboard Architecture

## Component Hierarchy (Simplified)
```
App.tsx
└── Routes
    ├── StormSignalView
    │   ├── DashboardSidebar
    │   ├── FeedPanel
    │   ├── MetricsPanel
    │   └── ClassificationPanel
    ├── ModelInformationDashboard
    │   ├── DashboardSidebar
    │   └── Model metrics + registry views
    └── AboutPage
        └── DashboardSidebar
```

## Data Flow
1. **User interaction** (filters, message input, navigation).
2. **React request** via `fetch` to Flask `/api/*` endpoints.
3. **Flask services** read data/model outputs and return JSON payloads.
4. **React state updates** via hooks (`useState`, `useEffect`).
5. **UI renders** updated feed, metrics, or classification results.

```
User → React UI → /api/* (Flask) → Model/Data services → JSON → React state → UI
```

## State Management
- Local component state and derived memoization (`useState`, `useEffect`, `useMemo`).
- No global state library; state is scoped to dashboard views.
- Error states are tracked per panel to keep the UI responsive.

## Routing Structure
- React Router handles SPA routes.
- Flask serves the same `app/static/dashboard/index.html` for each SPA route.
- Public routes:
  - `/dashboard`
  - `/production-model`
  - `/about`
- Legacy routes (redirected):
  - `/api/dashboard`
  - `/api/model-info-dashboard`
  - `/api/about`

## Build & Deployment
1. `npm run build` generates `dist/` assets.
2. `python scripts/build_dashboard.py` copies `dist/` to `app/static/dashboard/`.
3. Flask serves the static bundle and SPA shell for dashboard routes.

## Why It’s Built This Way
- **Single HTML shell** simplifies deployment: Flask serves static assets for all dashboard pages.
- **Relative API paths** keep the frontend environment-agnostic (local, staging, production).
- **Component-scoped state** reduces complexity and keeps UI code predictable.
