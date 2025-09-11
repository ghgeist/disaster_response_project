#!/usr/bin/env python3
"""
Enhanced model comparison tool for disaster response classification experiments.

This script provides crystal-clear comparison between different experiments,
making it easy for portfolio reviewers to understand model performance improvements.
"""

import sys
import os
import json
import pandas as pd
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from disasterproject.utils.experiment_tracker import ExperimentTracker


def load_experiment_results(experiment_name: str) -> Dict[str, Any]:
    """
    Load results from an experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary with experiment results
    """
    tracker = ExperimentTracker()
    experiment_info = tracker.get_experiment_info(experiment_name)
    
    if experiment_info is None:
        print(f"❌ Experiment '{experiment_name}' not found")
        return {}
    
    return experiment_info


def compare_experiments(experiment_names: List[str]) -> pd.DataFrame:
    """
    Compare multiple experiments and return a summary DataFrame.
    
    Args:
        experiment_names: List of experiment names to compare
        
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for exp_name in experiment_names:
        exp_info = load_experiment_results(exp_name)
        if exp_info:
            comparison_data.append({
                'experiment': exp_name,
                'sampling_method': exp_info.get('config', {}).get('sampling_method', 'unknown'),
                'created': exp_info.get('created', 'unknown'),
                'has_model': len(exp_info.get('models', [])) > 0,
                'has_results': 'results' in exp_info,
                'data_shape': exp_info.get('config', {}).get('data_shape', {}).get('X_train', 'unknown')
            })
    
    return pd.DataFrame(comparison_data)


def print_experiment_summary(experiment_name: str):
    """
    Print a detailed summary of a single experiment.
    
    Args:
        experiment_name: Name of the experiment
    """
    exp_info = load_experiment_results(experiment_name)
    
    if not exp_info:
        return
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*60}")
    
    print(f"📅 Created: {exp_info.get('created', 'Unknown')}")
    print(f"📁 Directory: {exp_info.get('directory', 'Unknown')}")
    
    # Configuration
    config = exp_info.get('config', {})
    if config:
        print(f"\n⚙️  Configuration:")
        print(f"   Sampling Method: {config.get('sampling_method', 'Unknown')}")
        print(f"   Test Size: {config.get('test_size', 'Unknown')}")
        print(f"   Random State: {config.get('random_state', 'Unknown')}")
        
        data_shape = config.get('data_shape', {})
        if data_shape:
            print(f"   Training Data Shape: {data_shape.get('X_train', 'Unknown')}")
            print(f"   Test Data Shape: {data_shape.get('X_test', 'Unknown')}")
    
    # Models
    models = exp_info.get('models', [])
    if models:
        print(f"\n🤖 Models:")
        for model in models:
            print(f"   - {model}")
    
    # Results
    results = exp_info.get('results', {})
    if results:
        print(f"\n📊 Results:")
        for key, value in results.items():
            print(f"   {key}: {value}")


def list_all_experiments():
    """List all available experiments."""
    tracker = ExperimentTracker()
    experiments = tracker.list_experiments()
    
    if not experiments:
        print("No experiments found.")
        return
    
    print(f"\n📋 Available Experiments ({len(experiments)}):")
    print("-" * 50)
    
    for i, exp in enumerate(experiments, 1):
        exp_info = tracker.get_experiment_info(exp)
        sampling_method = exp_info.get('config', {}).get('sampling_method', 'unknown') if exp_info else 'unknown'
        print(f"{i:2d}. {exp:<30} ({sampling_method})")


def main():
    """
    Main function with interactive comparison interface.
    """
    print("🔬 Disaster Response Classification - Model Comparison Tool")
    print("=" * 60)
    
    tracker = ExperimentTracker()
    
    while True:
        print("\nOptions:")
        print("1. List all experiments")
        print("2. Compare experiments")
        print("3. Show experiment details")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            list_all_experiments()
        
        elif choice == "2":
            experiments = tracker.list_experiments()
            if not experiments:
                print("No experiments found.")
                continue
            
            print(f"\nAvailable experiments: {', '.join(experiments)}")
            exp_input = input("Enter experiment names (comma-separated): ").strip()
            
            if exp_input:
                exp_names = [name.strip() for name in exp_input.split(',')]
                comparison_df = compare_experiments(exp_names)
                
                if not comparison_df.empty:
                    print(f"\n📊 Experiment Comparison:")
                    print(comparison_df.to_string(index=False))
                else:
                    print("No valid experiments found for comparison.")
        
        elif choice == "3":
            experiments = tracker.list_experiments()
            if not experiments:
                print("No experiments found.")
                continue
            
            print(f"\nAvailable experiments: {', '.join(experiments)}")
            exp_name = input("Enter experiment name: ").strip()
            
            if exp_name:
                print_experiment_summary(exp_name)
        
        elif choice == "4":
            print("Goodbye! 👋")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
