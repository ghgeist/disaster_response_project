#!/usr/bin/env python3
"""
Build and deploy the Model Information React dashboard to Flask static folder.

This script:
1. Builds the React app in _vendor/model_information_dashboard
2. Copies the build output to app/static/model_info_dashboard/

Usage:
    python scripts/build_model_info_dashboard.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
VENDOR_DIR = PROJECT_ROOT / "_vendor" / "model_information_dashboard"
DIST_DIR = VENDOR_DIR / "dist"
STATIC_DIR = PROJECT_ROOT / "app" / "static" / "model_info_dashboard"


def build_react_app():
    """Build the React app using npm."""
    print("Building Model Information dashboard...")
    try:
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=VENDOR_DIR,
            check=True,
            capture_output=True,
            text=True,
            shell=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print("\nBuild failed!", file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False
    except FileNotFoundError:
        print("Error: npm not found. Please install Node.js and npm.", file=sys.stderr)
        return False


def copy_build_output():
    """Copy dist/ contents to app/static/model_info_dashboard/."""
    if not DIST_DIR.exists():
        print(f"Error: Build output not found at {DIST_DIR}", file=sys.stderr)
        return False

    print(f"Copying build output to {STATIC_DIR}...")
    try:
        if STATIC_DIR.exists():
            shutil.rmtree(STATIC_DIR)
        shutil.copytree(DIST_DIR, STATIC_DIR)
        print("✓ Model Information dashboard deployed successfully!")
        return True
    except Exception as e:
        print(f"Error copying files: {e}", file=sys.stderr)
        return False


def main():
    if not build_react_app():
        sys.exit(1)
    if not copy_build_output():
        sys.exit(1)


if __name__ == "__main__":
    main()
