#!/usr/bin/env python3
"""Check which threshold files exist and which one the app will use."""

import json
from pathlib import Path

def main():
    model_dir = Path(__file__).parent.parent.parent / "model"
    print(f"Model directory: {model_dir}")
    print()
    
    # Find all threshold files
    threshold_files = list(model_dir.glob("*threshold*.json"))
    print("All threshold files found:")
    for tf in sorted(threshold_files):
        print(f"  - {tf.name}")
        try:
            with open(tf, 'r') as f:
                data = json.load(f)
                model_ref = data.get('metadata', {}).get('model', 'unknown')
                print(f"    References: {model_ref}")
        except Exception as e:
            print(f"    Error reading: {e}")
    print()
    
    # Check which one the app will use (simulate the logic)
    # Matches _find_production_thresholds_file in app/routes/api.py
    candidates = [
        f for f in model_dir.iterdir()
        if f.is_file() and f.name.endswith("_thresholds.json") and not f.name.startswith("optimized_")
    ]
    if candidates:
        thresholds_path = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"✅ App will use: {thresholds_path.name}")
        with open(thresholds_path, 'r') as f:
            data = json.load(f)
            model_ref = data.get('metadata', {}).get('model', 'unknown')
            print(f"   References: {model_ref}")
    else:
        print("❌ No threshold file found (app will return None)")
        # Check if optimized_* files exist but are ignored
        optimized_files = [
            f for f in model_dir.iterdir()
            if f.is_file() and f.name.endswith("_thresholds.json") and f.name.startswith("optimized_")
        ]
        if optimized_files:
            print(f"   Note: {len(optimized_files)} optimized_* threshold file(s) exist but are ignored by the app")
    
    # Check current model
    current_model = model_dir / "disaster_lr_v25-11-06_prod_2025-11-06.pkl"
    if current_model.exists():
        expected_thresholds = model_dir / f"{current_model.stem}_thresholds.json"
        print()
        print(f"Current model: {current_model.name}")
        print(f"Expected thresholds: {expected_thresholds.name}")
        print(f"Exists: {expected_thresholds.exists()}")

if __name__ == "__main__":
    main()
