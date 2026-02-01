#!/usr/bin/env python3
"""
Centralized experimental path management for disaster response project.

This module provides a unified interface for managing experimental artifact paths
in the experiments/experimental_runs/<date>/ structure.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentalArtifacts:
    """Container for experimental artifacts with their paths and metadata."""
    model_path: Optional[str] = None
    metrics_path: Optional[str] = None
    info_path: Optional[str] = None
    config_path: Optional[str] = None
    summary_path: Optional[str] = None
    base_dir: Optional[str] = None
    display_name: Optional[str] = None
    # Enhanced metadata
    model_name: Optional[str] = None
    experiment_date: Optional[str] = None
    experiment_type: Optional[str] = None
    hyperparameters: Optional[Dict] = None
    sampling_strategy: Optional[str] = None
    descriptive_name: Optional[str] = None


class ExperimentalPathManager:
    """
    Manages experimental artifact paths in experiments/experimental_runs/<date>/ structure.
    """

    def __init__(self):
        """Initialize path manager."""
        self.base = Path("experiments/experimental_runs")

    def get_latest_experimental_artifacts(self) -> Optional[ExperimentalArtifacts]:
        """
        Find the latest experimental artifacts.

        Returns:
            ExperimentalArtifacts object with paths, or None if no artifacts found
        """
        return self._find_artifacts()

    def _find_artifacts(self) -> Optional[ExperimentalArtifacts]:
        """Find artifacts in experiments/experimental_runs/<date>/ structure."""
        if not self.base.exists():
            return None

        # Find all date directories and check them for artifacts, starting with latest
        date_dirs = [
            d for d in self.base.iterdir()
            if d.is_dir() and self._is_date_directory(d)
        ]

        if not date_dirs:
            return None

        # Sort directories by date (newest first) and try each one
        sorted_dirs = sorted(date_dirs, key=lambda d: d.name, reverse=True)

        for directory in sorted_dirs:
            artifacts = self._extract_artifacts_from_dir(directory, f"experiments/experimental_runs/{directory.name}")
            if artifacts:  # Found artifacts in this directory
                return artifacts

        return None  # No artifacts found in any date directory

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
        date_pattern_alt = directory.name.replace('-', '_').replace('_', '-', 2) + '_'  # Handle 2025_09-16_ pattern
        potential_files.update({
            f'{date_pattern}_performance_metrics.csv': 'metrics_path',
            f'{date_pattern_alt}performance_metrics.csv': 'metrics_path',
            f'{directory.name}_training_log.json': 'summary_path',
            f'{directory.name}_hyperparameter_search_results.json': 'summary_path',
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

        # Enrich with metadata if we found artifacts
        if artifacts.metrics_path or artifacts.model_path:
            self._enrich_metadata(artifacts, directory)
            return artifacts

        return None

    def _enrich_metadata(self, artifacts: ExperimentalArtifacts, directory: Path) -> None:
        """Enrich artifacts with metadata extracted from files and directory structure."""
        # Extract date from directory name if it's a date directory
        if self._is_date_directory(directory):
            artifacts.experiment_date = directory.name

        # Extract model name from model file
        if artifacts.model_path:
            model_file = Path(artifacts.model_path)
            artifacts.model_name = model_file.stem

            # Parse experiment type from model filename
            artifacts.experiment_type = self._parse_experiment_type(model_file.name)

        # Load hyperparameters from various sources
        artifacts.hyperparameters = self._load_hyperparameters(directory)

        # Determine sampling strategy
        artifacts.sampling_strategy = self._determine_sampling_strategy(directory)

        # Create descriptive name
        artifacts.descriptive_name = self._create_descriptive_name(artifacts)

    def _parse_experiment_type(self, model_filename: str) -> str:
        """Parse experiment type from model filename."""
        filename_lower = model_filename.lower()

        if 'grid-search' in filename_lower or 'gridsearch' in filename_lower:
            if 'comprehensive' in filename_lower:
                return "Comprehensive Grid Search"
            elif 'original' in filename_lower:
                return "Original Grid Search"
            else:
                return "Grid Search"
        elif 'optimized' in filename_lower:
            return "Optimized Model"
        elif 'baseline' in filename_lower:
            return "Baseline Model"
        elif 'lightweight' in filename_lower:
            return "Lightweight Model"
        elif 'prod' in filename_lower or 'production' in filename_lower:
            return "Production Model"
        else:
            return "Experimental Model"

    def _load_hyperparameters(self, directory: Path) -> Optional[Dict]:
        """Load hyperparameters from various sources in the directory."""
        # Check for hyperparameter search results
        hyperparams_files = [
            f"{directory.name}_hyperparameter_search_results.json",
            "hyperparameter_search_results.json",
            f"{directory.name}_training_log.json",
            "training_log.json"
        ]

        for filename in hyperparams_files:
            filepath = directory / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Try different keys where hyperparameters might be stored
                        if 'best_params' in data:
                            return data['best_params']
                        elif 'hyperparameters' in data:
                            return data['hyperparameters']
                        elif 'parameters' in data:
                            return data['parameters']
                except (json.JSONDecodeError, OSError):
                    continue

        # Check parent directories for config files
        config_dir = Path("experiments/experimental_configs/hyperparameters")
        if config_dir.exists() and directory.name:
            config_file = config_dir / f"{directory.name}_comprehensive-grid-search.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

        return None

    def _determine_sampling_strategy(self, directory: Path) -> str:
        """Determine the sampling strategy used."""
        # Check training log for sampling info
        training_log_files = [
            f"{directory.name}_training_log.json",
            "training_log.json"
        ]

        for filename in training_log_files:
            filepath = directory / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'sampling_strategy' in data:
                            return data['sampling_strategy']
                        elif 'experiment_config' in data and 'sampling' in data['experiment_config']:
                            return data['experiment_config']['sampling']
                except (json.JSONDecodeError, OSError):
                    continue

        # Infer from directory structure or model name
        model_files = list(directory.glob("*.pkl"))
        if model_files:
            model_name = model_files[0].name.lower()
            if 'smote' in model_name:
                return "SMOTE"
            elif 'adasyn' in model_name:
                return "ADASYN"
            elif 'baseline' in model_name:
                return "No Sampling"

        return "Unknown"

    def _create_descriptive_name(self, artifacts: ExperimentalArtifacts) -> str:
        """Create a human-readable descriptive name for the experiment."""
        parts = []

        if artifacts.experiment_date:
            parts.append(artifacts.experiment_date)

        if artifacts.experiment_type:
            parts.append(artifacts.experiment_type)

        if artifacts.sampling_strategy and artifacts.sampling_strategy != "Unknown":
            if artifacts.sampling_strategy != "No Sampling":
                parts.append(f"({artifacts.sampling_strategy})")

        if artifacts.hyperparameters:
            # Add key hyperparameters if available
            hp_parts = []
            if 'clf__estimator__n_estimators' in artifacts.hyperparameters:
                hp_parts.append(f"n_est={artifacts.hyperparameters['clf__estimator__n_estimators']}")
            if 'clf__estimator__max_depth' in artifacts.hyperparameters:
                depth = artifacts.hyperparameters['clf__estimator__max_depth']
                hp_parts.append(f"depth={depth if depth else 'None'}")
            if 'vect__ngram_range' in artifacts.hyperparameters:
                ngram = artifacts.hyperparameters['vect__ngram_range']
                hp_parts.append(f"ngrams={ngram}")

            if hp_parts:
                parts.append(f"[{', '.join(hp_parts)}]")

        return " ".join(parts) if parts else artifacts.display_name or "Unknown Experiment"

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
        if not self.base.exists():
            return None

        # Search in all date directories
        for date_dir in self.base.iterdir():
            if date_dir.is_dir():
                # Exact match
                exact_path = date_dir / model_identifier
                if exact_path.exists():
                    return str(exact_path)

                # Partial match
                for model_file in date_dir.glob("*.pkl"):
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

        output_dir = self.base / date_str
        output_dir.mkdir(parents=True, exist_ok=True)

        return str(output_dir)

    def validate_structure(self) -> Dict[str, any]:
        """
        Validate the experimental directory structure.

        Returns:
            Dictionary with validation results
        """
        results = {
            'exists': self.base.exists(),
            'artifacts': 0,
            'issues': []
        }

        # Count artifacts
        if results['exists']:
            artifacts = self._find_artifacts()
            if artifacts:
                results['artifacts'] = 1

        if not results['exists']:
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
