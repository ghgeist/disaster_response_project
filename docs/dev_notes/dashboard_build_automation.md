# Dashboard Build Automation Options

This document outlines several options for automating the React dashboard build and deployment process.

## Problem

When the React app in `_vendor/figma_make` is rebuilt, the generated `index.html` references new hashed asset filenames. These files must be copied to `app/static/dashboard/` for Flask to serve them. Forgetting this step results in a blank page.

## Solution Options

### Option 1: Python Script (Recommended for Manual Builds)

**File**: `scripts/build_dashboard.py`

**Usage**:
```bash
# Build and deploy
python scripts/build_dashboard.py

# Watch mode (for active development)
python scripts/build_dashboard.py --watch
```

**Pros**:
- ✅ Single command to build and deploy
- ✅ Works cross-platform (Windows/Linux/Mac)
- ✅ Can be integrated into other scripts
- ✅ Watch mode available for development

**Cons**:
- ⚠️ Still requires manual execution
- ⚠️ Requires Node.js/npm installed

**When to use**: When you want a simple, reliable way to rebuild the dashboard before testing or deploying.

---

### Option 2: npm Script (Quick & Simple)

**Add to `_vendor/figma_make/package.json`**:
```json
{
  "scripts": {
    "build": "vite build",
    "build:deploy": "vite build && node ../../scripts/deploy-dashboard.js",
    "dev": "vite"
  }
}
```

**Create `scripts/deploy-dashboard.js`** (Node.js script to copy files)

**Usage**:
```bash
cd _vendor/figma_make
npm run build:deploy
```

**Pros**:
- ✅ Familiar npm workflow
- ✅ Can be combined with other npm scripts
- ✅ Fast execution

**Cons**:
- ⚠️ Requires Node.js script for cross-platform file copying
- ⚠️ Must be run from vendor directory

**When to use**: If you prefer npm workflows and are already in the vendor directory.

---

### Option 3: Automatic Check on Flask Startup (Optional)

**Modify `run.py`** to optionally check if build is needed:

```python
import os
from pathlib import Path

from app.app import create_app

def check_dashboard_build():
    """Optionally check if dashboard needs rebuilding."""
    if os.environ.get('AUTO_BUILD_DASHBOARD') == '1':
        dashboard_dir = Path('app/static/dashboard')
        vendor_dist = Path('_vendor/figma_make/dist')
        
        # Check if dist is newer than deployed files
        if vendor_dist.exists() and dashboard_dir.exists():
            dist_mtime = max(f.stat().st_mtime for f in vendor_dist.rglob('*') if f.is_file())
            dashboard_mtime = max(f.stat().st_mtime for f in dashboard_dir.rglob('*') if f.is_file())
            
            if dist_mtime > dashboard_mtime:
                print("⚠️  Dashboard build is newer than deployed files.")
                print("   Run: python scripts/build_dashboard.py")

if __name__ == '__main__':
    check_dashboard_build()
    app = create_app()
    # ... rest of run.py
```

**Usage**:
```bash
# Enable auto-check
AUTO_BUILD_DASHBOARD=1 python run.py

# Or disable (default)
python run.py
```

**Pros**:
- ✅ Warns you if build is out of sync
- ✅ Non-intrusive (only warns, doesn't auto-build)
- ✅ Can be enabled/disabled via env var

**Cons**:
- ⚠️ Adds small startup overhead
- ⚠️ Only warns, doesn't fix automatically

**When to use**: As a safety net to catch forgotten builds.

---

### Option 4: Watch Mode for Active Development

**Usage**:
```bash
# Terminal 1: Watch React app (rebuilds on changes)
python scripts/build_dashboard.py --watch

# Terminal 2: Run Flask app
python run.py
```

**Or use Vite dev server directly** (if you configure proxy):
```bash
cd _vendor/figma_make
npm run dev  # Runs on http://localhost:5173
```

**Pros**:
- ✅ Automatic rebuilds on file changes
- ✅ Fast feedback loop during development
- ✅ No manual steps needed

**Cons**:
- ⚠️ Requires two terminal windows
- ⚠️ Dev server approach needs proxy configuration

**When to use**: During active frontend development when you're making frequent changes.

---

### Option 5: Pre-commit Hook (Prevent Commits with Stale Build)

**Create `.git/hooks/pre-commit`** (or use pre-commit framework):

```bash
#!/bin/bash
# Check if dashboard build is out of sync
DASHBOARD_DIR="app/static/dashboard"
VENDOR_DIST="_vendor/figma_make/dist"

if [ -d "$VENDOR_DIST" ] && [ -d "$DASHBOARD_DIR" ]; then
    # Compare modification times
    if [ "$VENDOR_DIST" -nt "$DASHBOARD_DIR" ]; then
        echo "⚠️  Dashboard build is newer than deployed files!"
        echo "   Run: python scripts/build_dashboard.py"
        echo "   Commit anyway? (y/N)"
        read -r response
        if [ "$response" != "y" ]; then
            exit 1
        fi
    fi
fi
```

**Pros**:
- ✅ Prevents committing stale builds
- ✅ Catches issues before they reach CI

**Cons**:
- ⚠️ Can be annoying if you intentionally want to commit without rebuilding
- ⚠️ Requires git hook setup

**When to use**: If you want to enforce build sync before commits.

---

## Recommended Setup

**For most users**: Use **Option 1** (Python script) as your primary method:

```bash
# Before testing/deploying
python scripts/build_dashboard.py
```

**For active development**: Use **Option 4** (watch mode):

```bash
# Terminal 1
python scripts/build_dashboard.py --watch

# Terminal 2  
python run.py
```

**Optional safety net**: Add **Option 3** (startup check) to `run.py` to catch forgotten builds.

---

## Implementation Status

- ✅ Option 1: Python script (`scripts/build_dashboard.py`) - **IMPLEMENTED**
- ⏳ Option 2: npm script - Not implemented (can add if preferred)
- ⏳ Option 3: Startup check - Not implemented (can add if desired)
- ✅ Option 4: Watch mode - **AVAILABLE** via `--watch` flag
- ⏳ Option 5: Pre-commit hook - Not implemented (can add if desired)

---

## Future Enhancements

- **Auto-build on startup**: Optionally run build automatically if out of sync (adds startup delay)
- **CI/CD integration**: Ensure dashboard is built in CI before deployment
- **Build cache**: Skip rebuild if source files haven't changed
