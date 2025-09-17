#!/usr/bin/env python3
"""
Centralized experimental path management for disaster response project.

This module provides a unified interface for managing experimental artifact paths,
handling both legacy (experiments/results/) and new (experiments/experimental_runs/)
path structures with backward compatibility.
"""

import os
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExperimentalArtifacts:
    """Container for experimental artifacts with their paths."""
    model_path: Optional[str] = None
    metrics_path: Optional[str] = None
    info_path: Optional[str] = None
    config_path: Optional[str] = None
    summary_path: Optional[str] = None
    base_dir: Optional[str] = None
    display_name: Optional[str] = None


class ExperimentalPathManager:
    """
    Manages experimental artifact paths with backward compatibility.

    Handles both legacy structure (experiments/results/) and new structure
    (experiments/experimental_runs/<date>/) while providing migration utilities.
    """

    def __init__(self, prefer_new_structure: bool = True):
        """
        Initialize path manager.

        Args:
            prefer_new_structure: If True, prefer new path structure when both exist
        """
        self.prefer_new_structure = prefer_new_structure
        self.legacy_base = Path("experiments/results")
        self.new_base = Path("experiments/experimental_runs")

    def get_latest_experimental_artifacts(self) -> Optional[ExperimentalArtifacts]:
        """
        Find the latest experimental artifacts, checking both old and new structures.

        Returns:
            ExperimentalArtifacts object with paths, or None if no artifacts found
        """
        if self.prefer_new_structure:
            # Try new structure first
            artifacts = self._find_artifacts_new_structure()
            if artifacts:
                return artifacts
            # Fallback to legacy
            return self._find_artifacts_legacy_structure()
        else:
            # Try legacy first
            artifacts = self._find_artifacts_legacy_structure()
            if artifacts:
                return artifacts
            # Fallback to new
            return self._find_artifacts_new_structure()

    def _find_artifacts_new_structure(self) -> Optional[ExperimentalArtifacts]:
        """Find artifacts in new experiments/experimental_runs/<date>/ structure."""
        if not self.new_base.exists():
            return None

        # Find latest date directory
        date_dirs = [
            d for d in self.new_base.iterdir()
            if d.is_dir() and self._is_date_directory(d)
        ]

        if not date_dirs:
            return None

        latest_dir = max(date_dirs, key=lambda d: d.name)
        return self._extract_artifacts_from_dir(latest_dir, f"experiments/experimental_runs/{latest_dir.name}")

    def _find_artifacts_legacy_structure(self) -> Optional[ExperimentalArtifacts]:
        """Find artifacts in legacy experiments/results/ structure."""
        if not self.legacy_base.exists():
            return None

        # Check for direct files in results/
        direct_artifacts = self._extract_artifacts_from_dir(self.legacy_base, "experiments/results")
        if direct_artifacts and direct_artifacts.metrics_path:
            # Update display name to include the specific file for legacy direct files
            direct_artifacts.display_name = f"experiments/results/performance_metrics.csv"
            return direct_artifacts

        # Check for date subdirectories in results/
        date_dirs = [
            d for d in self.legacy_base.iterdir()
            if d.is_dir() and self._is_date_directory(d)
        ]

        if not date_dirs:
            return None

        latest_dir = max(date_dirs, key=lambda d: d.name)
        return self._extract_artifacts_from_dir(latest_dir, f"experiments/results/{latest_dir.name}")

    def _extract_artifacts_from_dir(self, directory: Path, display_name: str) -> Optional[ExperimentalArtifacts]:
        """Extract artifact paths from a directory."""
        artifacts = ExperimentalArtifacts(
            base_dir=str(directory),
            display_name=display_name
        )

        # Look for common artifact files (including different naming patterns)
        potential_files = {
            'performance_metrics.csv': 'metrics_path',
            'MODEL_INFO.json': 'info_path',
            'model_config.json': 'config_path',
            'training_summary.json': 'summary_path'
        }

        # Also check for date-prefixed versions
        date_pattern = directory.name.replace('-', '_')
        potential_files.update({
            f'{date_pattern}_performance_metrics.csv': 'metrics_path',
            f'{directory.name}_training_log.json': 'summary_path',
        })

        for filename, attr_name in potential_files.items():
            file_path = directory / filename
            if file_path.exists():
                setattr(artifacts, attr_name, str(file_path))

        # If standard files not found, use glob patterns for flexible naming
        if not artifacts.metrics_path:
            metrics_files = list(directory.glob("*performance_metrics.csv"))
            if metrics_files:
                artifacts.metrics_path = str(metrics_files[0])

        if not artifacts.summary_path:
            summary_files = list(directory.glob("*training_log.json"))
            if summary_files:
                artifacts.summary_path = str(summary_files[0])

        # Look for model files (.pkl)
        model_files = list(directory.glob("*.pkl"))
        if model_files:
            # Use the most recent model file
            latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
            artifacts.model_path = str(latest_model)

        # Return artifacts only if we found at least metrics or model
        if artifacts.metrics_path or artifacts.model_path:
            return artifacts

        return None

    def _is_date_directory(self, path: Path) -> bool:
        """Check if directory name looks like an ISO date (YYYY-MM-DD)."""
        try:
            datetime.strptime(path.name, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def find_specific_model(self, model_identifier: str) -> Optional[str]:
        """
        Find a specific model by name or partial match.

        Args:
            model_identifier: Model filename or partial identifier

        Returns:
            Full path to model file, or None if not found
        """
        search_paths = []

        # Add new structure paths
        if self.new_base.exists():
            for date_dir in self.new_base.iterdir():
                if date_dir.is_dir():
                    search_paths.append(date_dir)

        # Add legacy structure paths
        if self.legacy_base.exists():
            search_paths.append(self.legacy_base)
            for date_dir in self.legacy_base.iterdir():
                if date_dir.is_dir():
                    search_paths.append(date_dir)

        # Search for model
        for search_dir in search_paths:
            # Exact match
            exact_path = search_dir / model_identifier
            if exact_path.exists():
                return str(exact_path)

            # Partial match
            for model_file in search_dir.glob("*.pkl"):
                if model_identifier in model_file.name:
                    return str(model_file)

        return None

    def get_output_directory(self, date_str: Optional[str] = None) -> str:
        """
        Get the appropriate output directory for new experimental artifacts.

        Args:
            date_str: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Path to output directory (creates if needed)
        """
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        output_dir = self.new_base / date_str
        output_dir.mkdir(parents=True, exist_ok=True)

        return str(output_dir)

    def migrate_legacy_artifacts(self, dry_run: bool = False) -> List[Tuple[str, str]]:
        """
        Migrate artifacts from legacy to new structure.

        Args:
            dry_run: If True, only show what would be moved

        Returns:
            List of (source, destination) tuples for moved files
        """
        moves = []

        if not self.legacy_base.exists():
            logger.info("No legacy artifacts to migrate")
            return moves

        # Process direct files in experiments/results/
        for item in self.legacy_base.iterdir():
            if item.is_file():
                # Determine date from file modification time or current date
                file_date = datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d")
                dest_dir = self.new_base / file_date
                dest_path = dest_dir / item.name

                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    if not dest_path.exists():
                        shutil.move(str(item), str(dest_path))
                        logger.info(f"Moved {item} -> {dest_path}")

                moves.append((str(item), str(dest_path)))

        # Process date subdirectories in experiments/results/
        for date_dir in self.legacy_base.iterdir():
            if date_dir.is_dir() and self._is_date_directory(date_dir):
                dest_dir = self.new_base / date_dir.name

                if not dry_run:
                    if dest_dir.exists():
                        # Merge contents
                        for item in date_dir.iterdir():
                            dest_path = dest_dir / item.name
                            if not dest_path.exists():
                                shutil.move(str(item), str(dest_path))
                                moves.append((str(item), str(dest_path)))
                        # Remove empty source directory
                        if not any(date_dir.iterdir()):
                            date_dir.rmdir()
                    else:
                        shutil.move(str(date_dir), str(dest_dir))
                        moves.append((str(date_dir), str(dest_dir)))
                        logger.info(f"Moved directory {date_dir} -> {dest_dir}")
                else:
                    moves.append((str(date_dir), str(dest_dir)))

        return moves

    def validate_structure(self) -> Dict[str, any]:
        """
        Validate the experimental directory structure.

        Returns:
            Dictionary with validation results
        """
        results = {
            'legacy_exists': self.legacy_base.exists(),
            'new_exists': self.new_base.exists(),
            'legacy_artifacts': 0,
            'new_artifacts': 0,
            'issues': []
        }

        # Count legacy artifacts
        if results['legacy_exists']:
            legacy_artifacts = self._find_artifacts_legacy_structure()
            if legacy_artifacts:
                results['legacy_artifacts'] = 1

        # Count new artifacts
        if results['new_exists']:
            new_artifacts = self._find_artifacts_new_structure()
            if new_artifacts:
                results['new_artifacts'] = 1

        # Check for issues
        if results['legacy_artifacts'] > 0 and results['new_artifacts'] > 0:
            results['issues'].append("Both legacy and new artifacts exist - consider migration")

        if not results['legacy_exists'] and not results['new_exists']:
            results['issues'].append("No experimental artifacts found")

        return results


# Convenience functions for backward compatibility
def get_latest_experimental_model_path() -> Optional[str]:
    """Get path to latest experimental model (backward compatibility function)."""
    manager = ExperimentalPathManager()
    artifacts = manager.get_latest_experimental_artifacts()
    return artifacts.model_path if artifacts else None


def get_latest_experimental_metrics_path() -> Optional[str]:
    """Get path to latest experimental metrics (backward compatibility function)."""
    manager = ExperimentalPathManager()
    artifacts = manager.get_latest_experimental_artifacts()
    return artifacts.metrics_path if artifacts else None


def find_experimental_model(model_name: str) -> Optional[str]:
    """Find experimental model by name (backward compatibility function)."""
    manager = ExperimentalPathManager()
    return manager.find_specific_model(model_name)