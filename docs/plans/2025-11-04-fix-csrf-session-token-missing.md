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

### Phase 1: Ensure Session Initialization on Form Rendering
**File**: `app/routes.py`

**Changes**:
- Initialize session explicitly when rendering forms that require CSRF tokens
- Touch the session object to ensure Flask creates the session cookie
- Add session initialization before `form.hidden_tag()` is called

**Implementation**:
```python
@app.route('/')
@app.route('/index')
def index():
    """Main page displaying visualizations and message classification form."""
    try:
        # Initialize session to ensure cookie is set for CSRF token
        from flask import session
        session.permanent = True  # Make session persistent
        # Touch session to ensure it's created (lazy initialization)
        if 'csrf_token_init' not in session:
            session['csrf_token_init'] = True
        
        # Create form instance (now session exists for CSRF token)
        form = MessageForm()
        
        # ... rest of existing code ...
```

### Phase 2: Add Session Initialization Middleware
**File**: `app/app.py`

**Changes**:
- Add `before_request` handler to ensure session is initialized for all requests
- Only initialize if session doesn't exist (avoid unnecessary work)
- Log session initialization for debugging

**Implementation**:
```python
# In create_app(), after CSRFProtect initialization
@app.before_request
def ensure_session():
    """Ensure session is initialized for CSRF token support."""
    from flask import session
    if 'session_initialized' not in session:
        session.permanent = True
        session['session_initialized'] = True
        # Log only in debug mode to avoid noise
        if app.debug:
            app.logger.debug("Session initialized for CSRF support")
```

### Phase 3: Fix Session Cookie Configuration for Replit
**File**: `app/config.py`

**Changes**:
- Ensure proper cookie domain/path configuration
- Add explicit session cookie name if needed
- Verify HTTPS requirement for `SameSite=None`

**Implementation**:
```python
# Session / cookie settings
ALLOW_THIRD_PARTY_COOKIES = os.environ.get('ALLOW_THIRD_PARTY_COOKIES', '1') == '1'
if ALLOW_THIRD_PARTY_COOKIES:
    SESSION_COOKIE_SAMESITE = 'None'
    SESSION_COOKIE_SECURE = True  # Required for SameSite=None
    # Ensure session cookie is set with proper attributes
    SESSION_COOKIE_HTTPONLY = True
    # Set session to be permanent (expires based on PERMANENT_SESSION_LIFETIME)
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
else:
    SESSION_COOKIE_SAMESITE = os.environ.get('SESSION_COOKIE_SAMESITE', 'Lax')
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True

# Explicitly set session cookie name (optional, but helps with debugging)
SESSION_COOKIE_NAME = 'disaster_response_session'
```

### Phase 4: Enhance CSRF Error Handler Diagnostics
**File**: `app/app.py`

**Changes**:
- Add more detailed diagnostics when session cookie is missing
- Check if session was initialized in current request
- Log cookie attributes to help debug Replit-specific issues

**Implementation**:
```python
@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    """Handle CSRF errors with detailed diagnostics."""
    reason = getattr(e, 'reason', 'unknown')
    description = getattr(e, 'description', str(e))
    has_session_cookie = 'session' in request.cookies
    from flask import session
    
    # Check if session was initialized in Flask but cookie not sent
    session_exists = 'session_initialized' in session if hasattr(session, 'get') else False
    
    app.logger.warning(
        "CSRF error: %s; reason=%s; method=%s; path=%s; referrer=%s; origin=%s; "
        "has_session_cookie=%s; session_exists=%s; cookie_samesite=%s; cookie_secure=%s",
        description,
        reason,
        request.method,
        request.path,
        request.referrer,
        request.headers.get('Origin'),
        has_session_cookie,
        session_exists,
        current_app.config.get('SESSION_COOKIE_SAMESITE'),
        current_app.config.get('SESSION_COOKIE_SECURE'),
    )
    return render_template('error.html', message="Your session expired or the form is invalid. Please refresh and try again."), 400
```

### Phase 5: Add Session Validation Endpoint (Optional)
**File**: `app/routes.py`

**Changes**:
- Add debug endpoint to check session/cookie status (dev only)
- Help diagnose cookie issues in Replit environment

**Implementation**:
```python
@app.route('/debug/session', methods=['GET'])
def debug_session():
    """Debug endpoint to check session and cookie status (dev only)."""
    if not current_app.debug:
        abort(404)
    
    from flask import session, request
    return {
        'has_session_cookie': 'session' in request.cookies,
        'session_keys': list(session.keys()) if session else [],
        'cookie_headers': dict(request.headers) if hasattr(request, 'headers') else {},
        'config': {
            'SESSION_COOKIE_SAMESITE': current_app.config.get('SESSION_COOKIE_SAMESITE'),
            'SESSION_COOKIE_SECURE': current_app.config.get('SESSION_COOKIE_SECURE'),
            'SESSION_COOKIE_HTTPONLY': current_app.config.get('SESSION_COOKIE_HTTPONLY'),
        }
    }, 200
```

## Implementation Priority

1. **High Priority** (Fix immediately):
   - Phase 1: Session initialization on form rendering
   - Phase 2: Session initialization middleware

2. **Medium Priority** (Improve reliability):
   - Phase 3: Session cookie configuration fixes
   - Phase 4: Enhanced CSRF error diagnostics

3. **Low Priority** (Nice to have):
   - Phase 5: Debug endpoint for troubleshooting

## Testing Strategy

1. **Unit Tests**:
   - Test that session is initialized when rendering forms
   - Test that `before_request` handler creates session
   - Test CSRF token generation after session initialization

2. **Integration Tests**:
   - Test full flow: GET `/` -> extract CSRF token -> POST `/classify` with token
   - Verify session cookie is set on GET request
   - Verify session cookie is sent on POST request
   - Test with `SameSite=None` and `Secure=True` configuration

3. **Manual Testing**:
   - Test on Replit deployment
   - Verify session cookie appears in browser DevTools
   - Test form submission with CSRF token
   - Test in different browser contexts (top-level, iframe if applicable)

4. **Replit-Specific Testing**:
   - Deploy to Replit and verify session cookie is set
   - Check browser console for cookie warnings
   - Verify HTTPS is active (required for `SameSite=None`)
   - Test form submission from Replit domain

## Expected Outcome

After implementation:
- Session cookie is **always set** on GET requests that render forms
- Session is initialized before CSRF token generation
- CSRF validation succeeds on POST `/classify`
- Error logs show `has_session_cookie=True` when CSRF succeeds
- Form submissions work reliably on Replit deployment

## Known Limitations

1. **Browser Cookie Policies**:
   - Some browsers may block `SameSite=None` cookies even with `Secure=True`
   - Third-party cookie blocking (Safari, Firefox) may prevent cookies in iframes
   - Replit subdomain routing should work, but browser policies may interfere

2. **Replit Environment**:
   - HTTPS must be properly configured (check Replit settings)
   - Domain/subdomain routing must allow cookie setting
   - Load balancer or proxy may strip/modify cookie headers

3. **Iframe Contexts**:
   - If app is embedded in iframe, third-party cookie restrictions apply
   - May need to communicate with parent window or use alternative CSRF strategies

## Alternative Solutions (If Primary Fix Fails)

1. **Disable CSRF for Specific Routes** (Last Resort):
   - Only if session cookies cannot be set in deployment environment
   - Use `@csrf.exempt` decorator on problematic routes
   - **WARNING**: Reduces security, only use if absolutely necessary

2. **Use Alternative CSRF Strategy**:
   - Store CSRF token in database instead of session
   - Use header-based CSRF tokens (requires JavaScript)
   - Use double-submit cookie pattern

3. **Adjust Cookie Settings**:
   - Try `SameSite='Lax'` instead of `None` if third-party context isn't needed
   - Ensure `Secure=True` only when HTTPS is guaranteed
   - Test with different cookie configurations

## Notes

- Session initialization is critical - Flask-WTF requires an active session to store CSRF tokens
- The `before_request` handler ensures session exists before any route handler runs
- Replit deployment may require environment variable configuration for proper cookie settings
- Monitor logs for `has_session_cookie` status - if still False after fixes, investigate browser/network issues

