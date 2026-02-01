#!/usr/bin/env python3
"""Test runner that handles Python executable detection across environments.

This script provides a portable way to run pytest across different environments:
- Local development (with venv)
- Replit (managed environment)
- Cursor Web UI (limited runtime availability)
- CI/CD environments

Best practices:
- Prefers direct `pytest` invocation (recommended by pytest docs)
- Falls back to `python3 -m pytest` then `python -m pytest`
- Handles environment-specific requirements gracefully
"""
# Standard library imports
import os
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Local imports
from disasterproject.utils.env import check_venv_activation, is_venv_required  # noqa: E402


def find_pytest_command():
    """Find the best available pytest command for the current environment.

    Returns:
        List of command parts to execute pytest, e.g., ['pytest'] or ['python3', '-m', 'pytest']

    Raises:
        RuntimeError: If no pytest command can be found
    """
    # Strategy 1: Try direct pytest invocation (pytest docs recommend this)
    for cmd in ['pytest', 'pytest3']:
        try:
            result = subprocess.run(
                [cmd, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return [cmd]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    # Strategy 2: Try python3 -m pytest (common in web environments)
    for python_cmd in ['python3', 'python']:
        try:
            result = subprocess.run(
                [python_cmd, '-m', 'pytest', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return [python_cmd, '-m', 'pytest']
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    raise RuntimeError(
        "No pytest command found. Tried: pytest, pytest3, python3 -m pytest, python -m pytest. "
        "Please ensure pytest is installed and available in PATH."
    )


def check_environment():
    """Check environment and provide helpful warnings if needed."""
    warnings = []

    if is_venv_required() and not check_venv_activation():
        warnings.append(
            "WARNING: Virtual environment is required but not activated. "
            "Tests may fail due to missing dependencies."
        )

    return warnings


def main():
    """Main entry point for test runner."""
    # Check environment
    warnings = check_environment()
    for warning in warnings:
        print(warning, file=sys.stderr)

    # Find pytest command
    try:
        pytest_cmd = find_pytest_command()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Build full command with user-provided arguments
    cmd = pytest_cmd + sys.argv[1:]

    # Execute pytest
    if os.getenv('VERBOSE_TEST_RUNNER', '').lower() == 'true':
        print(f"Running: {' '.join(cmd)}", file=sys.stderr)

    sys.exit(subprocess.run(cmd).returncode)


if __name__ == '__main__':
    main()
