# Frontend Development Guide

## Local Development Setup
1. Install dependencies:
   ```bash
   npm install
   ```
2. Start the Vite dev server:
   ```bash
   npm run dev
   ```
3. Run Flask separately to serve APIs (required for live data):
   ```bash
   python run.py
   ```

## Testing Workflow
- Unit/component tests:
  ```bash
  npm test
  # or
  npm run test:run
  ```
- UI smoke checks: run Flask and load the dashboard in a browser to confirm API responses.

## Debugging Tips
- **API errors**: Open browser devtools → Network tab → inspect `/api/*` calls.
- **Blank screen**: Ensure Flask serves `app/static/dashboard/index.html` for SPA routes.
- **Stale UI after changes**: If using Flask (not Vite), rebuild and redeploy the dashboard assets.

## Hot Reload Behavior
- `npm run dev` provides instant hot reload for React changes.
- When using Flask’s static assets, hot reload is **not** available; you must rebuild.

## Build vs. Deploy
- **Rebuild only** when you want a production bundle:
  ```bash
  npm run build
  ```
- **Rebuild + deploy** when Flask should serve the updated dashboard:
  ```bash
  python ../../scripts/build_dashboard.py
  ```

## Common Issues & Solutions
- **`vite` not found**: Run `npm install` before `npm run build`.
- **404 on SPA route refresh**: Confirm Flask routes send `index.html` for `/dashboard`, `/production-model`, and `/about`.
- **API data not loading**: Verify Flask is running and `/api/metrics` returns JSON.

## When to Rebuild
- Rebuild and deploy whenever changes must be visible through Flask (`/dashboard`).
- Skip rebuild when working exclusively through the Vite dev server.
