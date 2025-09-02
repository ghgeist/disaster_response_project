"""
Experiment tracking and management utilities.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional


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
    
    def list_experiments(self) -> list:
        """
        List all available experiments.
        
        Returns:
            List of experiment names
        """
        if not os.path.exists(self.base_experiments_dir):
            return []
        
        experiments = []
        for item in os.listdir(self.base_experiments_dir):
            item_path = os.path.join(self.base_experiments_dir, item)
            if os.path.isdir(item_path):
                experiments.append(item)
        
        return sorted(experiments)
    
    def get_experiment_info(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary with experiment information or None if not found
        """
        experiment_dir = os.path.join(self.base_experiments_dir, experiment_name)
        if not os.path.exists(experiment_dir):
            return None
        
        info = {
            'name': experiment_name,
            'directory': experiment_dir,
            'created': datetime.fromtimestamp(os.path.getctime(experiment_dir)).isoformat()
        }
        
        # Check for config file
        config_path = os.path.join(experiment_dir, 'configs', 'experiment_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                info['config'] = config_data.get('config', {})
                info['timestamp'] = config_data.get('timestamp', '')
        
        # Check for results file
        results_path = os.path.join(experiment_dir, 'results', 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
                info['results'] = results_data.get('results', {})
        
        # Check for model files
        models_dir = os.path.join(experiment_dir, 'models')
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            info['models'] = model_files
        
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
