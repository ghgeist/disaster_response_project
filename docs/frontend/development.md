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

## Local UI + Replit Backend (SSH Tunnel)
Use this mode when the database and Flask app are running on Replit, but you want fast local UI iteration with Vite hot reload.

1. Start the backend on Replit (normal workflow).
2. Create an SSH tunnel from local port `5000` to the Replit app's `127.0.0.1:5000`:
   ```powershell
   ssh -i "$HOME\.ssh\replit" -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -p 22 -N -L 5000:127.0.0.1:5000 <replit-user>@<replit-host>
   ```
3. In a second terminal, run Vite locally from `_vendor/figma_make`:
   ```bash
   npm run dev
   ```
4. Open the dashboard at:
   - `http://localhost:5173/static/dashboard/`

### API proxy behavior in this mode
- Vite proxies `/api/*` requests to `http://127.0.0.1:5000`.
- The SSH tunnel forwards those requests to the Replit backend.
- This allows live API data with local frontend hot reload.

### Quick verification
- Confirm tunnel is active:
  ```powershell
  Test-NetConnection 127.0.0.1 -Port 5000
  ```
  Expected: `TcpTestSucceeded : True`
- Confirm API proxy path works by opening:
  - `http://localhost:5173/api/model-info`

### Troubleshooting
- **`ECONNREFUSED 127.0.0.1:5000` in Vite logs**: Tunnel is down or stuck at authentication; restart SSH command.
- **SSH prompts for password unexpectedly**: Ensure `-i` points to the correct private key file and keep `-o IdentitiesOnly=yes`.
- **Still seeing old errors after tunnel recovers**: Restart `npm run dev` and hard-refresh the browser.

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
