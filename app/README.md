# Flask Application

This directory contains the Flask layer for Storm Signal: application setup, configuration, routes, service wiring, templates, and static assets. The app loads the trained model and staged SQLite data, exposes classification and dashboard APIs, and serves the public React dashboard.

## Entry Points

- Local development: `python run.py`
- App factory: `app.app:create_app()`
- Production WSGI entry point: `wsgi.py`

Do not run `app/app.py` directly. The project uses `run.py` as the local entry point.

## Current Layout

```text
app/
├── __init__.py              # Package exports for create_app
├── app.py                   # Flask app factory and error handling
├── config.py                # Environment-based configuration
├── extensions.py            # Shared Flask extensions (CSRF)
├── forms.py                 # Flask-WTF forms
├── routes/                  # Blueprints
│   ├── __init__.py          # Blueprint registration
│   ├── api.py               # JSON endpoints for dashboard and model info
│   ├── classification.py    # /go and /classify flows
│   ├── health.py            # Health and diagnostics endpoints
│   └── home.py              # SPA shell routes and home redirects
├── services/                # Data/model/health services
├── static/                  # CSS, built dashboard assets, demo JSON
├── templates/               # Jinja templates for server-rendered pages
├── utils/                   # Environment validation, logging, helpers
└── visualizations.py        # Chart helpers for server-rendered views
```

## How The App Starts

`create_app()` in `app/app.py` is responsible for:

- Loading configuration from `app/config.py`
- Initializing logging
- Preparing NLTK resources
- Validating the runtime environment
- Registering CSRF protection
- Wiring services into the app
- Registering route blueprints
- Attaching global error handlers and security headers

If required inputs are missing, startup fails instead of serving a half-configured app.

## Required Runtime Inputs

The Flask app expects:

- A staged SQLite database at `data/02_stg/stg_disaster_response.db`
- A production model artifact in `model/`

By default, `app/config.py` auto-discovers the newest production model matching `disaster_*_prod_*.pkl`. You can override that with `MODEL_FILENAME`.

## Main Routes

Public pages:

- `GET /dashboard`
- `GET /production-model`
- `GET /about`

Server-rendered and classification routes:

- `GET /`
- `GET` and `POST /go`
- `GET` and `POST /classify`

Health and diagnostics:

- `GET /health`
- `GET /health/detailed`
- `GET /admin/model-health`
- `GET /api/model-health`
- `GET /api/performance-diagnostics`

Dashboard JSON APIs:

- `GET /api/feed`
- `GET /api/metrics`
- `GET /api/categories`
- `POST /api/classify`
- `GET /api/model-info`
- `GET /api/model-info/dashboard`

There are also legacy `/api/*` redirect routes for older dashboard paths.

## Frontend Integration

The React frontend lives in `_vendor/figma_make/`, but Flask serves the built assets from `app/static/dashboard/`.

Typical workflow after frontend changes:

```bash
cd _vendor/figma_make
npm run build
python ../../scripts/build_dashboard.py
```

`run.py` can warn when the built frontend is newer than the files currently deployed under `app/static/dashboard/`.

## Configuration

Useful environment variables:

- `FLASK_ENV` or `FLASK_DEBUG` for debug behavior
- `SECRET_KEY` for session and CSRF signing
- `HOST` and `PORT` for bind settings
- `LOG_LEVEL` for log verbosity
- `MODEL_FILENAME` to force a specific model artifact
- `ALLOW_THIRD_PARTY_COOKIES` for embedded or preview contexts

See `app/config.py` for the full configuration surface.

## Development Notes

- Add new HTTP behavior under the appropriate blueprint in `app/routes/`
- Put business logic in `app/services/` instead of route handlers when possible
- Keep environment checks, logging, and request helpers in `app/utils/`
- If you change frontend behavior, rebuild the dashboard assets Flask serves

## Troubleshooting

If the app fails during startup:

- Confirm the database exists at `data/02_stg/stg_disaster_response.db`
- Confirm there is a compatible production model in `model/`
- Check whether `MODEL_FILENAME` points to a file that actually exists

If API routes work but the dashboard looks stale:

- Rebuild the React app in `_vendor/figma_make/`
- Run `python scripts/build_dashboard.py`

If forms fail with CSRF errors in development:

- Refresh the page and retry
- Confirm cookies are enabled for your local session
- Check the app logs for the CSRF diagnostic messages added by `app/app.py`
