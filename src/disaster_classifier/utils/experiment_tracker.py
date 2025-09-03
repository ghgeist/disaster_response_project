"""
Experiment tracking and management utilities.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List


class ExperimentTracker:
    """
    Tracks and manages ML experiments with organized directory structure.
    """
    
    def __init__(self, base_experiments_dir: str = "experiments"):
        """
        Initialize the experiment tracker.
        
        Args:
            base_experiments_dir: Base directory for all experiments
        """
        self.base_experiments_dir = base_experiments_dir
        self.ensure_experiments_dir()
    
    def ensure_experiments_dir(self):
        """Ensure the experiments directory exists."""
        os.makedirs(self.base_experiments_dir, exist_ok=True)
        # Also ensure flat structure buckets
        for subdir in ['configs', 'models', 'results']:
            os.makedirs(os.path.join(self.base_experiments_dir, subdir), exist_ok=True)
    
    def create_experiment_dir(self, experiment_name: str) -> str:
        """
        Create a directory for a new experiment.
        
        Args:
            experiment_name: Name of the experiment (e.g., 'baseline_no_sampling')
            
        Returns:
            Path to the created experiment directory
        """
        experiment_dir = os.path.join(self.base_experiments_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create subdirectories for different types of outputs
        for subdir in ['models', 'results', 'configs', 'logs']:
            os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
        
        return experiment_dir
    
    def save_experiment_config(self, experiment_name: str, config: Dict[str, Any]) -> str:
        """
        Save experiment configuration to JSON file.
        
        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary
            
        Returns:
            Path to the saved config file
        """
        experiment_dir = self.create_experiment_dir(experiment_name)
        config_path = os.path.join(experiment_dir, 'configs', 'experiment_config.json')
        
        # Add metadata
        config_with_metadata = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': config
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_with_metadata, f, indent=2)
        
        logging.info(f"Experiment config saved to: {config_path}")
        return config_path

    # --- Flat structure helpers -------------------------------------------------
    def save_experiment_config_flat(self, slug: str, experiment_name: str, config: Dict[str, Any]) -> str:
        """
        Save experiment configuration using the flat 3-bucket layout.

        Args:
            slug: Unique identifier for the run (e.g., 'smote_v1-20250903T104500')
            experiment_name: Human-friendly experiment name (e.g., 'smote_conservative_v1')
            config: Configuration dictionary

        Returns:
            Path to the saved flat config file
        """
        self.ensure_experiments_dir()
        config_path = os.path.join(self.base_experiments_dir, 'configs', f"{slug}.json")

        config_with_metadata = {
            'slug': slug,
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': config,
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_with_metadata, f, indent=2)

        logging.info(f"[flat] Experiment config saved to: {config_path}")
        return config_path
    
    def save_model(self, experiment_name: str, model, model_filename: str = "model.pkl") -> str:
        """
        Save model to experiment directory.
        
        Args:
            experiment_name: Name of the experiment
            model: The trained model
            model_filename: Name of the model file
            
        Returns:
            Path to the saved model file
        """
        experiment_dir = self.create_experiment_dir(experiment_name)
        model_path = os.path.join(experiment_dir, 'models', model_filename)
        
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logging.info(f"Model saved to: {model_path}")
        return model_path

    def save_model_flat(self, slug: str, model, model_extension: str = "pkl") -> str:
        """
        Save model using the flat 3-bucket layout.

        Args:
            slug: Unique identifier for the run
            model: Trained model to serialize
            model_extension: File extension (default 'pkl')

        Returns:
            Path to the saved model file
        """
        self.ensure_experiments_dir()
        model_filename = f"{slug}.{model_extension}"
        model_path = os.path.join(self.base_experiments_dir, 'models', model_filename)

        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        logging.info(f"[flat] Model saved to: {model_path}")
        return model_path
    
    def save_results(self, experiment_name: str, results: Dict[str, Any], 
                    results_filename: str = "results.json") -> str:
        """
        Save experiment results to JSON file.
        
        Args:
            experiment_name: Name of the experiment
            results: Results dictionary
            results_filename: Name of the results file
            
        Returns:
            Path to the saved results file
        """
        experiment_dir = self.create_experiment_dir(experiment_name)
        results_path = os.path.join(experiment_dir, 'results', results_filename)
        
        # Add metadata
        results_with_metadata = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        logging.info(f"Results saved to: {results_path}")
        return results_path

    def save_results_flat(self, slug: str, results: Dict[str, Any]) -> str:
        """
        Save results JSON using the flat 3-bucket layout.

        Args:
            slug: Unique identifier for the run
            results: Results dictionary

        Returns:
            Path to the saved flat results file
        """
        self.ensure_experiments_dir()
        results_path = os.path.join(self.base_experiments_dir, 'results', f"{slug}_results.json")

        results_with_metadata = {
            'slug': slug,
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_metadata, f, indent=2)

        logging.info(f"[flat] Results saved to: {results_path}")
        return results_path
    
    def list_experiments(self) -> list:
        """
        List all available experiments.
        
        Returns:
            List of experiment names or slugs (supports legacy folders and flat layout)
        """
        if not os.path.exists(self.base_experiments_dir):
            return []

        experiments: List[str] = []

        # Legacy per-experiment directories
        for item in os.listdir(self.base_experiments_dir):
            item_path = os.path.join(self.base_experiments_dir, item)
            if os.path.isdir(item_path) and item not in {"configs", "models", "results"}:
                experiments.append(item)

        # Flat layout slugs (prefer results/*.json presence)
        results_dir = os.path.join(self.base_experiments_dir, 'results')
        if os.path.isdir(results_dir):
            for filename in os.listdir(results_dir):
                if filename.endswith('_results.json'):
                    slug = filename[:-len('_results.json')]
                    experiments.append(slug)

        # Also include any configs without results yet
        configs_dir = os.path.join(self.base_experiments_dir, 'configs')
        if os.path.isdir(configs_dir):
            for filename in os.listdir(configs_dir):
                if filename.endswith('.json'):
                    slug = filename[:-5]
                    experiments.append(slug)

        # Deduplicate and sort
        experiments = sorted(sorted(set(experiments)))
        return experiments
    
    def get_experiment_info(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary with experiment information or None if not found
        """
        experiment_dir = os.path.join(self.base_experiments_dir, experiment_name)

        # Legacy folder-based experiment
        if os.path.exists(experiment_dir) and os.path.isdir(experiment_dir):
            info = {
                'name': experiment_name,
                'directory': experiment_dir,
                'created': datetime.fromtimestamp(os.path.getctime(experiment_dir)).isoformat()
            }

            # Config
            config_path = os.path.join(experiment_dir, 'configs', 'experiment_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    info['config'] = config_data.get('config', {})
                    info['timestamp'] = config_data.get('timestamp', '')

            # Results
            results_path = os.path.join(experiment_dir, 'results', 'results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                    info['results'] = results_data.get('results', {})

            # Models
            models_dir = os.path.join(experiment_dir, 'models')
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
                info['models'] = model_files

            return info

        # Flat layout: treat experiment_name as slug
        slug = experiment_name
        base_dir = self.base_experiments_dir
        info = {
            'name': slug,
            'directory': base_dir,
        }

        # Config
        flat_config = os.path.join(base_dir, 'configs', f'{slug}.json')
        if os.path.exists(flat_config):
            with open(flat_config, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                info['config'] = cfg.get('config', {})
                info['timestamp'] = cfg.get('timestamp', '')
                info['experiment_name'] = cfg.get('experiment_name', slug)

        # Results
        flat_results = os.path.join(base_dir, 'results', f'{slug}_results.json')
        if os.path.exists(flat_results):
            with open(flat_results, 'r', encoding='utf-8') as f:
                res = json.load(f)
                info['results'] = res.get('results', {})

        # Model
        models_dir = os.path.join(base_dir, 'models')
        candidate = os.path.join(models_dir, f'{slug}.pkl')
        if os.path.exists(candidate):
            info['models'] = [os.path.basename(candidate)]
        else:
            info['models'] = []

        # Created time fallback: from config or result file
        if 'timestamp' in info and info['timestamp']:
            info['created'] = info['timestamp']
        else:
            for p in [flat_config, flat_results, models_dir]:
                if os.path.exists(p):
                    info['created'] = datetime.fromtimestamp(os.path.getctime(p)).isoformat()
                    break

        return info


def create_experiment_name(sampling_method: str, version: str = "v1") -> str:
    """
    Create a standardized experiment name.
    
    Args:
        sampling_method: Method used for sampling ('baseline', 'smote', 'adasyn', 'conservative')
        version: Version identifier
        
    Returns:
        Standardized experiment name
    """
    if sampling_method == 'baseline':
        return f"baseline_no_sampling_{version}"
    elif sampling_method == 'smote':
        return f"smote_conservative_{version}"
    elif sampling_method == 'adasyn':
        return f"adasyn_moderate_{version}"
    elif sampling_method == 'conservative':
        return f"conservative_sampling_{version}"
    else:
        return f"{sampling_method}_{version}"


def build_slug(sampling_method: str, version: str = "v1", run_id: Optional[str] = None) -> str:
    """
    Build a unique slug for a run using method, version, and a timestamp run id.

    Args:
        sampling_method: 'baseline' | 'smote' | 'adasyn' | 'conservative' | custom
        version: version tag (e.g., 'v1')
        run_id: optional timestamp string; default now in format YYYYMMDDTHHMMSS

    Returns:
        Slug string like 'smote_v1-20250903T104500'
    """
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%dT%H%M%S')
    method_part = sampling_method
    return f"{method_part}_{version}-{run_id}"
