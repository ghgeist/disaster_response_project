---
title: "Python Cache Management for Import Error Resolution"
date: "2025-09-02"
status: "accepted"
tags: ["python", "dependencies", "troubleshooting", "environment"]
author: "OpenAI Assistant"
related: []
---

# Python Cache Management for Import Error Resolution

**Date**: 2025-09-02  
**Status**: Accepted  
**Deciders**: Development Team  
**Tags**: python, dependencies, troubleshooting, environment

## Context

The Flask application was experiencing intermittent ImportError when importing numpy through pandas:

```
ImportError: Unable to import required dependencies:
numpy: Error importing numpy: you should not try to import numpy from
        its source directory; please exit the numpy source tree, and relaunch
        your python interpreter from there.
```

This error occurred despite numpy being properly installed in the project's isolated Python environment (`.pythonlibs/`). Investigation revealed the presence of compiled Python files (`.pyc`) from multiple Python versions (3.12 and 3.13) in the cache directories, causing import system conflicts.

## Decision

Implement a systematic Python cache cleaning approach as the primary solution for resolving import conflicts in the project environment:

1. **Immediate Resolution**: Clean all Python cache files when import errors occur
2. **Preventive Maintenance**: Establish cache cleaning as a standard troubleshooting step
3. **Documentation**: Document the cache cleaning process for future reference

### Implementation Commands:
```bash
# Remove all __pycache__ directories
find . -name "__pycache__" -type d -exec rm -rf {} +

# Remove all compiled Python files
find . -name "*.pyc" -delete
```

## Consequences

### Positive
- Resolves numpy/pandas import errors immediately and reliably
- Eliminates conflicts between Python versions in cache files
- Simple, fast solution that doesn't require environment rebuilding
- Preserves existing package installations and configurations
- Can be easily automated or scripted for future use

### Negative  
- Temporary performance impact on first import after cache clearing (modules need recompilation)
- Requires manual intervention when import errors occur
- Does not prevent the root cause of mixed version cache files

### Neutral
- Cache files are automatically regenerated during normal Python execution
- No impact on application functionality once cache is cleared
- Compatible with existing Cursor/Replit environment setup

## Follow-up (2025-09-02): System Library Dependency Issue  
Subsequent troubleshooting revealed another root cause for the same NumPy import error: the Replit container was missing system-level shared libraries that NumPy wheels depend on, specifically `libstdc++.so.6` (GNU C++ runtime) and `libz.so.1` (zlib).  
Clearing the Python byte-code cache provided only a temporary reprieve. The **permanent** resolution is to ensure these libraries are available in the active Nix profile.  
  
### Mitigation Steps  
1. Add the required libraries to the `[nix]` section of `.replit`:  
   ```toml
   [nix]
   channel = "stable-25_05"
   pkgs = [
     "stdenv.cc.cc.lib",  # provides libstdc++.so.6
     "zlib"               # provides libz.so.1
   ]
   ```  
2. Restart the workspace so the new Nix profile is activated (`kill 1` in the shell).  
3. Re-install NumPy/Pandas wheels so they link against the newly available libraries:  
   ```bash
   pip uninstall -y numpy pandas
   pip cache purge
   pip install numpy pandas
   ```  
  
### Outcome  
With the shared libraries present, NumPy and Pandas import cleanly, eliminating the misleading “source directory” error message. While this ADR still advocates cache cleaning as a first-line response, we capture the system-library lesson here for future incidents.  

## Alternatives Considered

1. **Virtual Environment Recreation**: Complete rebuild of Python environment
   - Rejected: Overkill for cache-related issues, time-consuming
   
2. **Package Reinstallation**: Reinstall numpy/pandas packages
   - Rejected: Packages were correctly installed; issue was cache-related
   
3. **Python Version Standardization**: Force single Python version
   - Rejected: Environment already uses consistent Python 3.12; mixed cache was legacy issue
   
4. **Environment Variables**: Modify PYTHONPATH or similar
   - Rejected: Would not address underlying cache conflicts

## References

- [Python Import System Documentation](https://docs.python.org/3/reference/import.html)
- [Numpy Installation Troubleshooting](https://numpy.org/doc/stable/user/troubleshooting-importerror.html)
- Flask Application Import Chain: `run.py` → `app.app` → `routes` → `services` → `pandas` → `numpy`
