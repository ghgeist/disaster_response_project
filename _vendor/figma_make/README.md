# Storm Signal React Dashboard

## Project Overview
The Storm Signal React dashboard is the single-page application (SPA) that powers the public-facing disaster response experience. It provides a real-time interface for classifying messages, monitoring category trends, and reviewing model health. The dashboard is served by the Flask backend, which exposes JSON APIs under `/api/*` and delivers the built assets from `app/static/dashboard/`.

## Quick Start
```bash
# Install dependencies
npm install

# Development (with hot reload)
npm run dev

# Build for production
npm run build

# Build and deploy to Flask static folder
python ../../scripts/build_dashboard.py
```

## Architecture Overview
- **Component structure:** `src/app/components/` contains dashboard UI, detail views, and shared UI primitives.
- **Key routes:** `/api/dashboard`, `/api/model-info-dashboard`, `/api/about` (SPA routes handled by React Router; Flask serves the same `index.html` shell).
- **Data flow:** React → Flask API endpoints (`/api/feed`, `/api/classify`, `/api/model-info`, etc.) → JSON response → UI updates.
- **Build output:** `npm run build` emits `dist/`, and `python ../../scripts/build_dashboard.py` copies that build to `app/static/dashboard/` so Flask can serve it.

## Key Components
- `StormSignalView.tsx` — Main dashboard layout with the live feed, metrics, and classification tooling.
- `ModelInformationDashboard.tsx` — Model performance, registry, and category health views.
- `ClassificationPanel.tsx` — Message classification form and results display.
- `FeedPanel.tsx` — Real-time signal feed with filters and severity cues.
- `MetricsPanel.tsx` — Category performance and volume metrics.

## Development Workflow
- **Local development:** `npm run dev` starts the Vite dev server with hot reload.
- **Testing changes:** `npm test` or `npm run test:run` executes Vitest.
- **Building:** `npm run build` writes the production build to `dist/`.
- **Deploying:** `python ../../scripts/build_dashboard.py` copies `dist/` into `app/static/dashboard/`.
- **Important:** After React changes, rebuild and deploy so Flask serves the updated static assets.

## API Integration
- The dashboard consumes Flask endpoints under `/api/*` using relative URLs (no separate API base URL required).
- Errors are handled in the UI with fallbacks or toast messaging to keep the dashboard responsive even if a backend request fails.

## Tech Stack
- React 18 + TypeScript
- Vite
- React Router
- Tailwind CSS
- Radix UI
- Vitest

## File Structure
```
src/
├── app/
│   ├── App.tsx              # Main router
│   ├── components/
│   │   ├── dashboard/       # Main dashboard components
│   │   ├── detail/          # Detail view components
│   │   └── ui/              # Reusable UI components
│   ├── data/                # API clients and data types
│   └── utils/               # Utility functions
└── styles/                  # Global styles
```

## Troubleshooting
- **No UI changes after editing components:** run `npm run build` and `python ../../scripts/build_dashboard.py` to redeploy the static assets.
- **API errors in the UI:** confirm Flask is running and that `/api/*` endpoints respond locally.
- **Blank page on refresh:** ensure Flask is serving `app/static/dashboard/index.html` for SPA routes.
