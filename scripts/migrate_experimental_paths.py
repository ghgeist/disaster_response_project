#!/usr/bin/env python3
"""
Migration utility for experimental path structure.

Migrates artifacts from legacy experiments/results/ to new experiments/experimental_runs/<date>/
structure while maintaining data integrity and providing validation.

Usage:
    python scripts/migrate_experimental_paths.py --dry-run  # Preview changes
    python scripts/migrate_experimental_paths.py           # Execute migration
    python scripts/migrate_experimental_paths.py --validate # Validate structure
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from disasterproject.utils.experimental_paths import ExperimentalPathManager


def main():
    parser = argparse.ArgumentParser(description="Migrate experimental artifacts to new path structure")
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be migrated without making changes')
    parser.add_argument('--validate', action='store_true',
                       help='Validate current experimental structure')
    parser.add_argument('--force', action='store_true',
                       help='Force migration even if target exists')

    args = parser.parse_args()

    manager = ExperimentalPathManager()

    if args.validate:
        print("🔍 VALIDATING EXPERIMENTAL STRUCTURE")
        print("=" * 50)

        results = manager.validate_structure()

        print(f"Legacy structure exists: {'✅' if results['legacy_exists'] else '❌'}")
        print(f"New structure exists: {'✅' if results['new_exists'] else '❌'}")
        print(f"Legacy artifacts found: {results['legacy_artifacts']}")
        print(f"New artifacts found: {results['new_artifacts']}")

        if results['issues']:
            print("\n⚠️  ISSUES DETECTED:")
            for issue in results['issues']:
                print(f"  - {issue}")
        else:
            print("\n✅ No issues detected")

        # Show current artifacts
        print(f"\n📄 CURRENT ARTIFACTS:")
        artifacts = manager.get_latest_experimental_artifacts()
        if artifacts:
            print(f"  Display name: {artifacts.display_name}")
            print(f"  Base directory: {artifacts.base_dir}")
            if artifacts.model_path:
                print(f"  Model: {artifacts.model_path}")
            if artifacts.metrics_path:
                print(f"  Metrics: {artifacts.metrics_path}")
            if artifacts.info_path:
                print(f"  Info: {artifacts.info_path}")
        else:
            print("  No artifacts found")

        return

    print("🚀 EXPERIMENTAL PATH MIGRATION")
    print("=" * 50)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print()

    # Perform migration
    moves = manager.migrate_legacy_artifacts(dry_run=args.dry_run)

    if not moves:
        print("✅ No artifacts need migration")
        return

    print(f"📦 {'Planned' if args.dry_run else 'Completed'} migrations:")
    for source, dest in moves:
        print(f"  {source} -> {dest}")

    if args.dry_run:
        print(f"\n💡 Run without --dry-run to execute these {len(moves)} migrations")
    else:
        print(f"\n✅ Successfully migrated {len(moves)} items")

        # Validate after migration
        print("\n🔍 POST-MIGRATION VALIDATION:")
        results = manager.validate_structure()

        if results['issues']:
            print("⚠️  Issues detected after migration:")
            for issue in results['issues']:
                print(f"  - {issue}")
        else:
            print("✅ Migration completed successfully")


if __name__ == "__main__":
    main()