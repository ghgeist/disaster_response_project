# Plan: Fix CSRF Session Token Missing Error

**Date**: 2025-11-04  
**Issue**: CSRF validation failing with "The CSRF session token is missing" on POST `/classify`  
**Root Cause**: Session cookie not being set (`has_session_cookie=False`), preventing Flask-WTF from storing/retrieving CSRF tokens

## Problem Analysis

### Observed Error
```
[2025-11-04 14:59:22,865] WARNING in app: CSRF error: The CSRF session token is missing.; 
reason=unknown; method=POST; path=/classify; 
referrer=https://7a9b561d-f544-467f-b104-566f909f7948-00-1araapd9be0s4.riker.replit.dev/; 
origin=https://7a9b561d-f544-467f-b104-566f909f7948-00-1araapd9be0s4.riker.replit.dev; 
has_session_cookie=False
```

### Root Causes Identified

1. **Flask Session Lazy Initialization** (`app/routes.py`)
   - Flask sessions are lazy - they only get created when something is written to `session`
   - When `form.hidden_tag()` renders the CSRF token on GET `/`, Flask-WTF tries to store the token in the session
   - If session hasn't been touched/initialized, the session cookie may not be set properly
   - The session cookie must exist before CSRF token can be stored/validated

2. **Session Cookie Configuration Issues** (`app/config.py`)
   - When `ALLOW_THIRD_PARTY_COOKIES = '1'` (default), config sets:
     - `SESSION_COOKIE_SAMESITE = 'None'`
     - `SESSION_COOKIE_SECURE = True`
   - For `SameSite=None`, cookies **require** `Secure=True` and HTTPS
   - On Replit, the domain is `*.riker.replit.dev` which should support HTTPS
   - However, if session isn't initialized, no cookie is sent, so these settings don't matter

3. **Missing Session Initialization Guard**
   - No explicit session initialization before form rendering
   - No `before_request` handler to ensure session exists
   - Form rendering happens before session is guaranteed to exist

4. **Replit-Specific Cookie Behavior**
   - Replit uses subdomain routing (`*.riker.replit.dev`)
   - Browser may block cookies if they're not set properly on first request
   - Cross-origin or iframe contexts may prevent cookie setting (even with `SameSite=None`)

## Solution Plan

### Single Fix: Ensure Session Initialization
**File**: `app/app.py`

**Root Cause**: Flask sessions are lazy - they only initialize when written to. Flask-WTF needs an active session to store CSRF tokens, but if the session cookie was never set, CSRF validation fails.

**Solution**: Add a simple `before_request` handler to ensure session exists for all requests.

**Implementation**:
```python
# In create_app(), after CSRFProtect initialization (around line 121)
@app.before_request
def ensure_session():
    """Ensure session is initialized for CSRF token support."""
    from flask import session
    # Touch session to initialize it (Flask sessions are lazy)
    session.permanent = True
    if 'init' not in session:
        session['init'] = True
```

**Why this works**:
- Flask-WTF stores CSRF tokens in the session
- Writing to `session` ensures Flask creates the session cookie
- `before_request` runs before any route handler, so session exists when forms render
- Minimal code change, covers all routes automatically

## Testing Strategy

1. **Quick Manual Test**:
   - Deploy to Replit
   - Open app in browser, check DevTools → Application → Cookies → verify `session` cookie exists
   - Submit form on `/classify` - should work without CSRF error

2. **Existing Smoke Test**:
   - Run `pytest tests/test_csrf_smoke.py` - should pass
   - This test already validates the CSRF flow end-to-end

## Expected Outcome

After implementation:
- Session cookie is **always set** on GET requests that render forms
- Session is initialized before CSRF token generation
- CSRF validation succeeds on POST `/classify`
- Error logs show `has_session_cookie=True` when CSRF succeeds
- Form submissions work reliably on Replit deployment

## Notes

- Cookie configuration (`SameSite=None`, `Secure=True`) is already set in `app/config.py` for Replit
- This fix ensures the session cookie is actually created and sent
- If cookies still don't work after this fix, check Replit HTTPS configuration and browser console for cookie warnings

