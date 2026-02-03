#!/usr/bin/env python3
"""
Build and deploy the React dashboard to Flask static folder.

This script:
1. Builds the React app in _vendor/figma_make
2. Copies the build output to app/static/dashboard/

Usage:
    python scripts/build_dashboard.py
    python scripts/build_dashboard.py --watch  # Watch mode for development
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
VENDOR_DIR = PROJECT_ROOT / "_vendor" / "figma_make"
DIST_DIR = VENDOR_DIR / "dist"
STATIC_DASHBOARD_DIR = PROJECT_ROOT / "app" / "static" / "dashboard"


def build_react_app():
    """Build the React app using npm."""
    print("Building React app...")
    try:
        # Use shell=True on Windows to find npm in PATH
        # On Unix-like systems, shell=True also works fine
        result = subprocess.run(
            "npm run build",
            cwd=VENDOR_DIR,
            check=True,
            capture_output=True,
            text=True,
            shell=True,  # Required on Windows to find npm.cmd
        )
        # Print stdout first, then stderr (warnings)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nBuild failed!", file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        return False
    except FileNotFoundError:
        print("Error: npm not found. Please install Node.js and npm.", file=sys.stderr)
        return False


def copy_build_output():
    """Copy dist/ contents to app/static/dashboard/."""
    if not DIST_DIR.exists():
        print(f"Error: Build output not found at {DIST_DIR}", file=sys.stderr)
        return False

    print(f"Copying build output to {STATIC_DASHBOARD_DIR}...")
    try:
        # Remove existing dashboard directory contents
        if STATIC_DASHBOARD_DIR.exists():
            shutil.rmtree(STATIC_DASHBOARD_DIR)

        # Copy dist contents
        shutil.copytree(DIST_DIR, STATIC_DASHBOARD_DIR)
        print("✓ Dashboard build deployed successfully!")
        return True
    except Exception as e:
        print(f"Error copying files: {e}", file=sys.stderr)
        return False


def watch_mode():
    """Run Vite in watch mode (for active development)."""
    print("Starting Vite watch mode...")
    print("Note: This will rebuild automatically on file changes.")
    print("Press Ctrl+C to stop.")
    try:
        subprocess.run(["npm", "run", "dev"], cwd=VENDOR_DIR, shell=True)
    except KeyboardInterrupt:
        print("\nWatch mode stopped.")
    except FileNotFoundError:
        print("Error: npm not found. Please install Node.js and npm.", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build and deploy React dashboard")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run Vite in watch mode for development (does not copy to static)",
    )
    args = parser.parse_args()

    if args.watch:
        watch_mode()
        return

    # Build and copy
    if not build_react_app():
        sys.exit(1)

    if not copy_build_output():
        sys.exit(1)


if __name__ == "__main__":
    main()
