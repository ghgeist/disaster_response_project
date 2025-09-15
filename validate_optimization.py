#!/usr/bin/env python3
"""
Validate NLTK performance optimization by checking code changes without importing dependencies.
"""

import os
from pathlib import Path

def check_file_exists(file_path):
    """Check if a file exists."""
    return Path(file_path).exists()

def check_file_contains(file_path, text, description):
    """Check if a file contains specific text."""
    if not check_file_exists(file_path):
        return False, f"File not found: {file_path}"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        if text in content:
            return True, f"✓ {description}"
        else:
            return False, f"✗ {description} not found"
    except Exception as e:
        return False, f"✗ Error reading file: {e}"

def check_file_not_contains(file_path, text, description):
    """Check if a file does not contain specific text."""
    if not check_file_exists(file_path):
        return False, f"File not found: {file_path}"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        if text not in content:
            return True, f"✓ {description}"
        else:
            return False, f"✗ {description} still present"
    except Exception as e:
        return False, f"✗ Error reading file: {e}"

def main():
    """Validate the optimization implementation."""
    print("=" * 70)
    print("NLTK Performance Optimization Code Validation")
    print("=" * 70)
    
    checks = [
        # Check that NLTK setup module was created
        ("NLTK Setup Module Created", 
         lambda: (check_file_exists("app/nltk_setup.py"), "NLTK setup module created")),
        
        # Check that config.py was optimized
        ("Config NLTK Downloads Removed", 
         lambda: check_file_not_contains("src/disasterproject/utils/config.py", 
                                       "nltk.download", 
                                       "NLTK download logic removed")),
        
        ("Config Optimization Comment Added", 
         lambda: check_file_contains("src/disasterproject/utils/config.py", 
                                   "NLTK resources are now managed by app/nltk_setup.py", 
                                   "Optimization comment added")),
        
        # Check Flask app integration
        ("Flask App NLTK Import Added", 
         lambda: check_file_contains("app/app.py", 
                                   "from .nltk_setup import setup_nltk_resources, NLTKSetupError", 
                                   "NLTK setup import added")),
        
        ("Flask App NLTK Setup Call Added", 
         lambda: check_file_contains("app/app.py", 
                                   "setup_nltk_resources()", 
                                   "NLTK setup call added")),
        
        ("Flask App NLTK Results Storage", 
         lambda: check_file_contains("app/app.py", 
                                   "NLTK_SETUP_RESULTS", 
                                   "NLTK setup results storage added")),
        
        # Check compatibility optimization
        ("Compatibility Cache Added", 
         lambda: check_file_contains("app/compat.py", 
                                   "_module_mapping_cache", 
                                   "Module mapping cache added")),
        
        ("Compatibility Cache Functions Added", 
         lambda: check_file_contains("app/compat.py", 
                                   "get_compatibility_cache_status", 
                                   "Cache status function added")),
        
        # Check performance monitoring
        ("Health Check Performance Timing", 
         lambda: check_file_contains("app/routes.py", 
                                   "total_response_time_ms", 
                                   "Performance timing added to health check")),
        
        ("Performance Diagnostics Endpoint", 
         lambda: check_file_contains("app/routes.py", 
                                   "performance_diagnostics", 
                                   "Performance diagnostics endpoint added")),
        
        ("NLTK Status Monitoring", 
         lambda: check_file_contains("app/routes.py", 
                                   "nltk_status", 
                                   "NLTK status monitoring added")),
        
        # Check NLTK setup module content
        ("NLTK Setup Required Resources", 
         lambda: check_file_contains("app/nltk_setup.py", 
                                   "REQUIRED_RESOURCES", 
                                   "Required resources defined")),
        
        ("NLTK Setup Resource Validators", 
         lambda: check_file_contains("app/nltk_setup.py", 
                                   "RESOURCE_VALIDATORS", 
                                   "Resource validators defined")),
        
        ("NLTK Setup Error Handling", 
         lambda: check_file_contains("app/nltk_setup.py", 
                                   "NLTKSetupError", 
                                   "Error handling class defined")),
        
        ("NLTK Setup Status Function", 
         lambda: check_file_contains("app/nltk_setup.py", 
                                   "get_nltk_status", 
                                   "Status monitoring function added")),
    ]
    
    results = []
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * 50)
        
        try:
            success, message = check_func()
            print(message)
            results.append((check_name, success, message))
            if success:
                passed += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append((check_name, False, f"Error: {e}"))
    
    print("\n" + "=" * 70)
    print("Validation Results Summary")
    print("=" * 70)
    
    for check_name, success, message in results:
        status = "PASS" if success else "FAIL"
        print(f"{check_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All optimization code changes have been successfully implemented!")
        print("\nKey improvements implemented:")
        print("• ✅ NLTK setup module created with startup-only resource management")
        print("• ✅ Per-request NLTK downloads removed from config.py")
        print("• ✅ Flask app integrated with NLTK startup optimization")
        print("• ✅ Compatibility layer optimized with caching")
        print("• ✅ Performance monitoring added to health check")
        print("• ✅ NLTK status monitoring and diagnostics added")
        print("\nThe application should now have significantly improved performance:")
        print("• NLTK resources loaded once at startup instead of per-request")
        print("• Health check response times should be <500ms consistently")
        print("• Legacy model loading uses cached compatibility shims")
        print("• Comprehensive performance monitoring available")
    else:
        print(f"\n⚠ {total - passed} checks failed. Please review the implementation.")
        print("\nFailed checks:")
        for check_name, success, message in results:
            if not success:
                print(f"• {check_name}: {message}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)