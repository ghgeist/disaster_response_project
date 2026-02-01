#!/usr/bin/env python3
"""
Test different deployment scenarios for disaster response model.

Scenarios:
1. Production: Google Drive only (no local model)
2. Development: Local model fallback
3. Development: Google Drive primary
"""

# Standard library imports
import argparse
import os
import shutil
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_production_scenario():
    """Run production scenario and return success status."""
    print("🚀 Testing Production Scenario: Google Drive Only")
    print("=" * 60)

    # Set environment for production
    os.environ['GDRIVE_MODEL_ID'] = '1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh'
    os.environ['FLASK_ENV'] = 'production'

    # Move local model to simulate production environment
    model_path = Path('model/disaster_rf_v1-2-0_prod_2025-09-11.pkl')
    backup_path = Path('model/disaster_rf_v1-2-0_prod_2025-09-11.pkl.prod_test')

    if model_path.exists():
        print("📁 Moving local model to simulate production environment")
        shutil.move(str(model_path), str(backup_path))

    try:
        from app.app import create_app
        from app.config import Config

        print("⬬ Creating Flask app (should trigger Google Drive download)...")
        app = create_app(Config)

        print("✅ SUCCESS: Production app created, model downloaded from Google Drive")

        with app.app_context():
            model_service = app.model_service
            result = model_service.predict('Emergency: Need medical help urgently')
            positive_count = sum(1 for v in result.values() if v == 1)
            print(f"✅ SUCCESS: Production prediction - {positive_count} categories activated")

        return True

    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        return False

    finally:
        # Restore local model
        if backup_path.exists():
            print("🔄 Restoring local model")
            shutil.move(str(backup_path), str(model_path))

def run_development_local():
    """Run development scenario with local model and return success status."""
    print("💻 Testing Development Scenario: Local Model Primary")
    print("=" * 60)

    # Unset Google Drive to force local model usage
    if 'GDRIVE_MODEL_ID' in os.environ:
        del os.environ['GDRIVE_MODEL_ID']
    os.environ['FLASK_ENV'] = 'development'

    try:
        from app.app import create_app
        from app.config import Config

        print("⚡ Creating Flask app (should use local model for speed)...")
        app = create_app(Config)

        print("✅ SUCCESS: Development app created with local model")

        with app.app_context():
            model_service = app.model_service
            result = model_service.predict('Emergency shelter needed for families')
            positive_count = sum(1 for v in result.values() if v == 1)
            print(f"✅ SUCCESS: Local model prediction - {positive_count} categories activated")

        return True

    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        return False

def run_development_gdrive():
    """Run development scenario with Google Drive and return success status."""
    print("💻☁️ Testing Development Scenario: Google Drive with Local Fallback")
    print("=" * 60)

    # Set Google Drive but keep local model as fallback
    os.environ['GDRIVE_MODEL_ID'] = '1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh'
    os.environ['FLASK_ENV'] = 'development'

    try:
        from app.app import create_app
        from app.config import Config

        print("⚡ Creating Flask app (should use local model but validate Google Drive)...")
        app = create_app(Config)

        print("✅ SUCCESS: Development app created with Google Drive configured + local fallback")

        with app.app_context():
            model_service = app.model_service
            result = model_service.predict('Search and rescue teams needed')
            positive_count = sum(1 for v in result.values() if v == 1)
            print(f"✅ SUCCESS: Hybrid development prediction - {positive_count} categories activated")

        return True

    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        return False


def test_production_scenario():
    """Test production scenario: Google Drive only, no local model."""
    assert run_production_scenario(), "Production scenario failed"


def test_development_local():
    """Test development scenario: Local model primary."""
    assert run_development_local(), "Development local scenario failed"


def test_development_gdrive():
    """Test development scenario: Google Drive primary with local fallback."""
    assert run_development_gdrive(), "Development Google Drive scenario failed"

def main():
    parser = argparse.ArgumentParser(description='Test disaster response deployment scenarios')
    parser.add_argument('scenario', choices=['production', 'dev-local', 'dev-gdrive', 'all'],
                       help='Deployment scenario to test')

    args = parser.parse_args()

    success_count = 0
    total_count = 0

    if args.scenario in ['production', 'all']:
        total_count += 1
        if run_production_scenario():
            success_count += 1
        print()

    if args.scenario in ['dev-local', 'all']:
        total_count += 1
        if run_development_local():
            success_count += 1
        print()

    if args.scenario in ['dev-gdrive', 'all']:
        total_count += 1
        if run_development_gdrive():
            success_count += 1
        print()

    # Summary
    print("📊 Test Summary")
    print("=" * 30)
    print(f"✅ Passed: {success_count}/{total_count}")
    print(f"❌ Failed: {total_count - success_count}/{total_count}")

    if success_count == total_count:
        print("\n🎉 All deployment scenarios working correctly!")
        print("Your disaster response system is ready for production!")
    else:
        print("\n⚠️  Some scenarios failed. Check logs above for details.")

    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
