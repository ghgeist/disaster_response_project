#!/usr/bin/env python
"""Pre-execution environment validation for ML experiments."""
import os
import sys
import shutil
import importlib.util
import pandas as pd
from pathlib import Path

def validate_environment():
    """Run all pre-execution checks."""
    checks = {}
    
    # Critical files
    checks['database_exists'] = os.path.exists('data/02_stg/stg_disaster_response.db')
    checks['production_model_exists'] = os.path.exists('model/disaster_rf_v25-09-16_prod_2025-09-19.pkl')
    
    # Environment
    checks['venv_active'] = sys.prefix != sys.base_prefix
    checks['python_version'] = sys.version_info >= (3, 12)
    
    # Resources (adjusted for local execution)
    disk_free_gb = shutil.disk_usage('.').free / (1024**3)
    checks['disk_space_gb'] = disk_free_gb > 20
    print(f"Disk space available: {disk_free_gb:.1f}GB (need 20GB+)")
    
    # Memory check (critical for local execution)
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        # Minimum 8GB total, but warn if less than 16GB
        checks['memory_available_gb'] = total_ram_gb >= 8
        
        print(f"Total RAM: {total_ram_gb:.1f}GB, Available: {available_ram_gb:.1f}GB")
        if total_ram_gb < 16:
            print("⚠️ WARNING: Less than 16GB RAM - may be slow or fail on large models")
            print("   Recommend: Skip Increment 4 (RandomForest) if attempted")
        if available_ram_gb < 4:
            print("⚠️ WARNING: Less than 4GB available RAM - close other applications")
    except ImportError:
        print("WARNING: psutil not available, skipping memory check")
        print("   Install with: pip install psutil")
        checks['memory_available_gb'] = True  # Assume OK but risky
    
    # Required scripts
    checks['required_scripts'] = all([
        os.path.exists('scripts/03_create_experimental_model.py'),
        os.path.exists('scripts/compare_models.py'),
        os.path.exists('src/disasterproject/models/pipeline.py')
    ])
    
    # Required packages
    checks['required_packages'] = all([
        importlib.util.find_spec(pkg) is not None 
        for pkg in ['sklearn', 'pandas', 'numpy', 'joblib', 'sqlalchemy']
    ])
    
    # Report results
    print("\n=== Pre-Execution Validation ===")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}: {passed}")
    
    failed = [k for k, v in checks.items() if not v]
    if failed:
        print(f"\n❌ VALIDATION FAILED: {failed}")
        print("Fix these issues before proceeding.")
        sys.exit(1)
    
    print("\n✅ All validation checks passed")
    return checks

def create_eval_split():
    """Create consistent evaluation split if missing."""
    eval_csv = 'data/04_fct/eval_ids.csv'
    eval_json = 'experiments/experimental_configs/eval_sets/eval_ids.json'
    
    if os.path.exists(eval_csv):
        print(f"✓ Eval split exists: {eval_csv}")
        return eval_csv
    
    if os.path.exists(eval_json):
        print(f"✓ Eval split exists: {eval_json}")
        return eval_json
    
    print("❌ No eval split found - creating one...")
    
    # Load data
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///data/02_stg/stg_disaster_response.db')
    df = pd.read_sql_query('SELECT id FROM messages', engine)
    
    # Create 80/20 split with fixed seed
    from sklearn.model_selection import train_test_split
    train_ids, eval_ids = train_test_split(
        df['id'].values, 
        test_size=0.2, 
        random_state=42
    )
    
    # Save eval IDs
    os.makedirs('data/04_fct', exist_ok=True)
    pd.DataFrame({'id': eval_ids}).to_csv(eval_csv, index=False)
    print(f"✓ Created eval split: {eval_csv} ({len(eval_ids)} samples)")
    
    return eval_csv

def backup_production_model():
    """Backup current production model."""
    prod_model = 'model/disaster_rf_v25-09-16_prod_2025-09-19.pkl'
    if not os.path.exists(prod_model):
        print("WARNING: Production model not found, skipping backup")
        return
    
    from datetime import datetime
    backup_dir = f"model/backups/{datetime.now().strftime('%Y-%m-%d')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    backup_path = os.path.join(backup_dir, os.path.basename(prod_model))
    shutil.copy2(prod_model, backup_path)
    print(f"✓ Backed up production model to: {backup_path}")

def test_baseline_loadable():
    """Verify production model can be loaded."""
    import joblib
    prod_model = 'model/disaster_rf_v25-09-16_prod_2025-09-19.pkl'
    
    try:
        model = joblib.load(prod_model)
        print(f"✓ Production model loadable: {type(model)}")
        return True
    except Exception as e:
        print(f"❌ Failed to load production model: {e}")
        return False

def create_experiment_directory():
    """Create experiment directory for tonight's run."""
    from datetime import datetime
    exp_dir = f"experiments/experimental_runs/{datetime.now().strftime('%Y-%m-%d')}"
    os.makedirs(exp_dir, exist_ok=True)
    print(f"✓ Experiment directory: {exp_dir}")
    return exp_dir

if __name__ == '__main__':
    print("=== ML Execution Environment Validation ===\n")
    
    # Run all checks
    validate_environment()
    eval_path = create_eval_split()
    backup_production_model()
    baseline_ok = test_baseline_loadable()
    exp_dir = create_experiment_directory()
    
    if not baseline_ok:
        print("\n❌ STOP: Production baseline not loadable")
        sys.exit(1)
    
    print("\n✅ Pre-execution validation complete - ready to proceed")
    print(f"   Eval set: {eval_path}")
    print(f"   Experiment dir: {exp_dir}")

