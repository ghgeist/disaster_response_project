---
title: "Debug Agent: CSRF Failures on POST /go"
date: "2025-09-12"
status: "active"
tags: ["documentation", "debugging", "security", "csrf"]
author: "runner"
related: ["docs/agents/debug-agent.md", "app/app.py", "app/routes.py", "app/templates/home.html", "app/config.py"]
---


# Debug Agent: CSRF Failures on POST /go

**Date**: 2025-09-12  
**Status**: Active  
**Priority**: High  
**Estimated Duration**: 30–60 min  
**Tags**: csrf, security, flask, wtf

## 🎯 Objective

Eliminate intermittent CSRF 400 errors on POST `/go` by ensuring the CSRF token is rendered, transmitted, and validated reliably in development and deployment environments.

## 📋 Success Criteria

- [ ] GET `/` renders a form containing `<input name="csrf_token" ...>`
- [ ] Programmatic smoke test can GET `/`, extract `csrf_token`, POST `/go`, and receive HTTP 200
- [ ] CSRFError handler logs actionable diagnostics (reason, method, path, referrer, origin, cookie presence)
- [ ] No spurious 400 CSRF errors during normal form submission in dev and demo sessions

## 🔍 Context

- Problem observed: periodic `400 Bad Request: The CSRF session token is missing.` when posting to `/go`.
- Recent refactor separated templates; only `home.html` renders the form and includes `{{ form.hidden_tag() }}` which should emit the CSRF token.
- CSRF protection is enabled globally via `CSRFProtect(app)` and `WTF_CSRF_ENABLED=True`.
- `WTF_CSRF_TIME_LIMIT=3600` (1h) in config. SECRET_KEY uses env var with a dev default locally.
- Added CSRF error handler in `app/app.py` to log request context and present a friendly message.

## 📝 Requirements

### Functional Requirements
- Reliable CSRF validation for POST `/go` via the rendered form on `/`.
- Smoke test to validate end-to-end token acquisition and submission.

### Technical Requirements  
- Maintain global CSRF protection; only pages with forms render tokens.
- Error handler must not leak sensitive data to the user; detailed logs go to server logs.

### Quality Requirements
- Minimal changes to user-facing templates; focus on robustness and observability.
- Keep changes small and reversible.

## 🛠️ Approach

1) Instrumentation: Ensure `CSRFError` handler logs reason, method, path, referrer, origin, and whether a session cookie was present.
2) Verification: Add a smoke test that GETs `/`, scrapes `csrf_token`, POSTs `/go` with the session cookie, expects 200.
3) Environment hygiene: Use a stable `SECRET_KEY` in deployed/demo environments; extend CSRF time limit in dev-only if long demos are expected.

## 📊 Acceptance Criteria

- Smoke test reliably passes on local dev environment and CI (where applicable).
- Manual submission via the home page succeeds consistently in a demo session.
- Logs show clear diagnostics if a CSRF failure occurs (no silent 400s).

## 🔗 Related Work

- `docs/agents/debug-agent.md` – process and methodology
- `app/templates/home.html` – the form and `{{ form.hidden_tag() }}`
- `app/routes.py` – `/` and `/go` route logic
- `app/app.py` – CSRFProtect and CSRFError handler
- `app/config.py` – CSRF and SECRET_KEY configuration

## 📈 Metrics

- Metric 1: Smoke test pass rate (aim: 100%)
- Metric 2: Count of CSRF 400 errors during a 30-minute demo (aim: 0)

## 🚨 Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Proxy/CDN stripping cookies | High | Low-Med | Validate cookie presence in logs; adjust proxy config; set SameSite=None; Secure as needed |
| SECRET_KEY rotation mid-session | Med | Low | Ensure stable env var in deployment; avoid restarts during demo |
| CSRF timeout during long demo | Low | Med | Increase `WTF_CSRF_TIME_LIMIT` in dev or set to `None` |

## 📄 Deliverables

- [ ] CSRF smoke test (GET `/` -> POST `/go`) in `tests/`
- [x] CSRF error handler with diagnostics in `app/app.py`
- [ ] Optional: dev-only `WTF_CSRF_TIME_LIMIT=None` toggle for demos

## Next Steps (Actionable)

1. Implement smoke test `tests/test_csrf_smoke.py` to GET `/`, extract `csrf_token`, POST `/go`, expect 200.
2. For demo stability (optional, dev only): set `WTF_CSRF_TIME_LIMIT=None` to disable token expiry during long sessions.
3. Ensure `SECRET_KEY` provided via environment in deployed/demo setups so sessions persist.
4. If issues persist behind proxies, validate SameSite/cookie attributes and confirm cookies are not stripped.


