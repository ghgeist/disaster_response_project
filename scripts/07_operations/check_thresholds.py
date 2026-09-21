#!/usr/bin/env python3
"""Check which threshold files exist and which one the app will use."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import Config  # noqa: E402
from app.routes.api import (  # noqa: E402
    _find_production_thresholds_file,
    _resolve_active_production_model_path,
)


def main() -> None:
    model_dir = REPO_ROOT / "model"
    print(f"Model directory: {model_dir}")
    print()

    threshold_files = list(model_dir.glob("*threshold*.json"))
    print("All threshold files found:")
    for tf in sorted(threshold_files):
        print(f"  - {tf.name}")
        try:
            with open(tf, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                model_ref = data.get("metadata", {}).get("model", "unknown")
                print(f"    metadata.model (training-source provenance): {model_ref}")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"    Error reading: {exc}")
    print()

    app = Flask(__name__)
    app.config["MODEL_PATH"] = Config.MODEL_PATH
    with app.app_context():
        active = _resolve_active_production_model_path(model_dir)
        thresholds_path = _find_production_thresholds_file(model_dir)

    if active is not None:
        print(f"Active production model: {active.name}")
        print(f"Expected stem-bound thresholds: {active.stem}_thresholds.json")
    else:
        print("Active production model: <none>")

    if thresholds_path is not None:
        print(f"✅ App will use: {thresholds_path.name}")
        with open(thresholds_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            model_ref = data.get("metadata", {}).get("model", "unknown")
            print(f"   metadata.model: {model_ref}")
    else:
        print("❌ No stem-bound threshold file found (app will return None)")
        optimized_files = [
            path
            for path in model_dir.iterdir()
            if path.is_file()
            and path.name.endswith("_thresholds.json")
            and path.name.startswith("optimized_")
        ]
        if optimized_files:
            print(
                f"   Note: {len(optimized_files)} optimized_* threshold file(s) "
                "exist but are ignored by the app"
            )


if __name__ == "__main__":
    main()
