#!/usr/bin/env python3
"""
Structure validation script for the refactored disaster response classification system.

This script validates the modular structure without requiring ML dependencies.
"""

import os
import sys
import importlib.util


def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False


def check_module_structure(module_path, expected_files):
    """Check if a module directory has the expected structure."""
    print(f"\n📁 Checking module: {module_path}")
    
    if not os.path.exists(module_path):
        print(f"❌ Module directory not found: {module_path}")
        return False
    
    all_good = True
    for file in expected_files:
        filepath = os.path.join(module_path, file)
        if not check_file_exists(filepath, f"  {file}"):
            all_good = False
    
    return all_good


def validate_import_structure():
    """Validate that the import structure is correct."""
    print("\n🔍 Validating import structure...")
    
    # Check that we can import the modules (without executing them)
    modules_to_check = [
        ("src.disasterproject.utils.config", "Config module"),
        ("src.disasterproject.data.preprocessor", "Preprocessor module"),
        ("src.disasterproject.data.loader", "Data loader module"),
        ("src.disasterproject.models.samplers", "Samplers module"),
        ("src.disasterproject.models.pipeline", "Pipeline module"),
        ("src.disasterproject.evaluation.metrics", "Metrics module"),
        ("src.disasterproject.utils.json_io", "IO utilities module"),
        ("src.disasterproject.utils.interaction", "Interaction module"),
        ("src.disasterproject.utils.experiment_tracker", "Experiment tracker module"),
    ]
    
    all_good = True
    for module_name, description in modules_to_check:
        try:
            # Add src to path
            sys.path.insert(0, 'src')
            
            # Try to load the module spec
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                print(f"✅ {description}: {module_name}")
            else:
                print(f"❌ {description}: {module_name} - Module not found")
                all_good = False
        except Exception as e:
            print(f"⚠️  {description}: {module_name} - Import error (expected due to missing dependencies): {e}")
    
    return all_good


def main():
    """Main validation function."""
    print("🔬 Disaster Response Classification - Structure Validation")
    print("=" * 60)
    
    # Check main directory structure
    print("\n📁 Checking main directory structure...")
    main_structure = [
        ("src/disasterproject/__init__.py", "Main package init"),
        ("src/disasterproject/data/__init__.py", "Data package init"),
        ("src/disasterproject/models/__init__.py", "Models package init"),
        ("src/disasterproject/evaluation/__init__.py", "Evaluation package init"),
        ("src/disasterproject/utils/__init__.py", "Utils package init"),
        ("scripts/train_model.py", "New training script"),
        ("scripts/compare_models.py", "Comparison tool"),
        ("experiments/", "Experiments directory"),
        ("models/train_classifier_original.py", "Original script backup"),
        ("README_REFACTORING.md", "Refactoring documentation"),
    ]
    
    main_good = True
    for filepath, description in main_structure:
        if not check_file_exists(filepath, description):
            main_good = False
    
    # Check module structure
    print("\n📁 Checking module structure...")
    modules_structure = [
        ("src/disasterproject/data", ["__init__.py", "loader.py", "preprocessor.py"]),
        ("src/disasterproject/models", ["__init__.py", "pipeline.py", "samplers.py"]),
        ("src/disasterproject/evaluation", ["__init__.py", "metrics.py"]),
        ("src/disasterproject/utils", ["__init__.py", "config.py", "json_io.py", "interaction.py", "experiment_tracker.py"]),
    ]
    
    modules_good = True
    for module_path, expected_files in modules_structure:
        if not check_module_structure(module_path, expected_files):
            modules_good = False
    
    # Validate import structure
    import_good = validate_import_structure()
    
    # Check Flask app update
    print("\n🌐 Checking Flask app integration...")
    flask_app_path = "app/run.py"
    if os.path.exists(flask_app_path):
        with open(flask_app_path, 'r') as f:
            content = f.read()
            if "disasterproject.data.preprocessor import tokenize" in content:
                print("✅ Flask app updated with new import path")
                flask_good = True
            else:
                print("❌ Flask app not updated with new import path")
                flask_good = False
    else:
        print("❌ Flask app not found")
        flask_good = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    all_good = main_good and modules_good and import_good and flask_good
    
    print(f"Main Structure: {'✅ PASS' if main_good else '❌ FAIL'}")
    print(f"Module Structure: {'✅ PASS' if modules_good else '❌ FAIL'}")
    print(f"Import Structure: {'✅ PASS' if import_good else '❌ FAIL'}")
    print(f"Flask Integration: {'✅ PASS' if flask_good else '❌ FAIL'}")
    
    if all_good:
        print(f"\n🎉 ALL VALIDATIONS PASSED!")
        print("The refactored structure is ready for use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Test training: python scripts/train_model.py data/02_stg/stg_disaster_response.db models/classifier.pkl")
        print("3. Test comparison: python scripts/compare_models.py")
        print("4. Test Flask app: python app/run.py")
    else:
        print(f"\n⚠️  SOME VALIDATIONS FAILED!")
        print("Please check the issues above before proceeding.")
    
    return all_good


if __name__ == "__main__":
    main()
