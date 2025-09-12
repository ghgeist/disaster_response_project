#!/usr/bin/env python3
"""
Model Naming Utility for Disaster Response ML Models

Implements standardized naming convention:
{domain}_{algorithm}_{version}_{environment}_{date}.pkl

Usage:
    python scripts/model_naming_utility.py --rename-current
    python scripts/model_naming_utility.py --generate-name --algorithm rf --version 1.2.0
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple


class ModelNamingUtility:
    """Utility for standardized ML model naming and management."""
    
    ALGORITHMS = {
        'rf': 'RandomForest',
        'lr': 'LogisticRegression', 
        'tfidf': 'TF-IDF + LogisticRegression',
        'xgb': 'XGBoost',
        'bert': 'BERT-based',
        'ensemble': 'Ensemble'
    }
    
    ENVIRONMENTS = {
        'prod': 'Production',
        'stg': 'Staging/UAT',
        'dev': 'Development', 
        'exp': 'Experimental'
    }
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.model_dir = self.base_dir / "model"
        self.experiments_dir = self.base_dir / "experiments"
    
    def generate_model_name(
        self, 
        algorithm: str,
        version: str,
        environment: str = 'prod',
        domain: str = 'disaster',
        date: str = None
    ) -> str:
        """
        Generate standardized model name.
        
        Args:
            algorithm: ML algorithm (rf, lr, tfidf, xgb, bert)
            version: Semantic version (e.g., '1.2.0')
            environment: Target environment (prod, stg, dev, exp)
            domain: Business domain (disaster, emergency, crisis)
            date: Training date (YYYY-MM-DD) or None for today
            
        Returns:
            Standardized model filename
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Algorithm must be one of: {list(self.ALGORITHMS.keys())}")
        
        if environment not in self.ENVIRONMENTS:
            raise ValueError(f"Environment must be one of: {list(self.ENVIRONMENTS.keys())}")
        
        # Convert version dots to hyphens for filename safety
        version_safe = version.replace('.', '-')
        
        # Use today's date if not provided
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        return f"{domain}_{algorithm}_v{version_safe}_{environment}_{date}.pkl"
    
    def parse_model_name(self, filename: str) -> Dict[str, str]:
        """
        Parse standardized model name into components.
        
        Returns:
            Dictionary with domain, algorithm, version, environment, date
        """
        try:
            base_name = filename.replace('.pkl', '')
            parts = base_name.split('_')
            
            if len(parts) != 5:
                return {'error': f'Invalid format: {filename}'}
            
            domain, algorithm, version_part, environment, date = parts
            version = version_part.replace('v', '').replace('-', '.')
            
            return {
                'domain': domain,
                'algorithm': algorithm,
                'algorithm_desc': self.ALGORITHMS.get(algorithm, 'Unknown'),
                'version': version,
                'environment': environment,
                'environment_desc': self.ENVIRONMENTS.get(environment, 'Unknown'),
                'date': date,
                'valid': True
            }
        except Exception as e:
            return {'error': f'Parse error: {e}', 'valid': False}
    
    def suggest_rename_current_model(self) -> Tuple[str, str]:
        """
        Suggest standardized name for current production model.
        
        Returns:
            Tuple of (current_path, suggested_name)
        """
        current_model = self.model_dir / "classifier.pkl"
        
        if not current_model.exists():
            raise FileNotFoundError(f"Current model not found: {current_model}")
        
        # Read model metadata to determine algorithm
        metadata_path = self.model_dir / "model_info.json"
        algorithm = 'rf'  # Default
        
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    # Detect algorithm from model info
                    if 'rf_params' in metadata:
                        algorithm = 'rf'
                    elif any('logistic' in str(v).lower() for v in metadata.values()):
                        algorithm = 'lr'
            except Exception:
                pass
        
        # Get file modification date
        mod_time = datetime.fromtimestamp(current_model.stat().st_mtime)
        date_str = mod_time.strftime('%Y-%m-%d')
        
        suggested_name = self.generate_model_name(
            algorithm=algorithm,
            version='1.2.0',  # Increment from your current version
            environment='prod',
            date=date_str
        )
        
        return str(current_model), suggested_name
    
    def rename_model_with_artifacts(self, old_path: str, new_name: str, dry_run: bool = True):
        """
        Rename model and all associated artifacts.
        
        Args:
            old_path: Current model file path
            new_name: New standardized name (without .pkl)
            dry_run: If True, only show what would be renamed
        """
        old_model_path = Path(old_path)
        new_base_name = new_name.replace('.pkl', '')
        model_dir = old_model_path.parent
        
        # Find all related files
        old_base_name = old_model_path.stem  # 'classifier' from 'classifier.pkl'
        related_files = []
        
        # Main model file
        related_files.append((
            old_model_path,
            model_dir / f"{new_base_name}.pkl"
        ))
        
        # Associated JSON/CSV files - check both prefixed and standalone versions
        artifact_mappings = [
            ('thresholds.json', '_thresholds.json'),
            ('label_order.json', '_labels.json'),
            ('model_info.json', '_metadata.json'),
            ('training_log.json', '_training.json'),
            ('performance_metrics.csv', '_metrics.csv'),
            ('class_weights.json', '_weights.json'),
            ('parameters.json', '_params.json'),
        ]
        
        for old_suffix, new_suffix in artifact_mappings:
            old_artifact = model_dir / old_suffix  # Current naming (no prefix)
                
            if old_artifact.exists():
                new_artifact = model_dir / f"{new_base_name}{new_suffix}"
                related_files.append((old_artifact, new_artifact))
        
        print(f"{'DRY RUN: ' if dry_run else ''}Renaming model and artifacts:")
        print("=" * 60)
        
        for old_file, new_file in related_files:
            print(f"{'[WOULD RENAME]' if dry_run else '[RENAMING]'} {old_file.name} -> {new_file.name}")
            
            if not dry_run:
                try:
                    shutil.move(str(old_file), str(new_file))
                    print(f"  ✅ Success")
                except Exception as e:
                    print(f"  ❌ Error: {e}")
        
        if dry_run:
            print(f"\nRun with --execute to perform actual renaming")
        else:
            print(f"\n✅ Model renamed to: {new_base_name}.pkl")


def main():
    parser = argparse.ArgumentParser(
        description='Model naming utility for disaster response ML models'
    )
    
    parser.add_argument('--rename-current', action='store_true',
                       help='Suggest standardized name for current production model')
    
    parser.add_argument('--generate-name', action='store_true',
                       help='Generate standardized name for new model')
    
    parser.add_argument('--algorithm', choices=['rf', 'lr', 'tfidf', 'xgb', 'bert'],
                       help='ML algorithm type')
    
    parser.add_argument('--version', help='Semantic version (e.g., 1.2.0)')
    
    parser.add_argument('--environment', choices=['prod', 'stg', 'dev', 'exp'],
                       default='prod', help='Target environment')
    
    parser.add_argument('--execute', action='store_true',
                       help='Execute the rename operation (default is dry run)')
    
    args = parser.parse_args()
    
    utility = ModelNamingUtility()
    
    if args.rename_current:
        try:
            current_path, suggested_name = utility.suggest_rename_current_model()
            print("Current Production Model Rename Suggestion")
            print("=" * 50)
            print(f"Current: {Path(current_path).name}")
            print(f"Suggested: {suggested_name}")
            
            # Parse the suggested name to show details
            details = utility.parse_model_name(suggested_name)
            if details.get('valid'):
                print(f"\nDetails:")
                print(f"  Algorithm: {details['algorithm_desc']}")
                print(f"  Version: {details['version']}")
                print(f"  Environment: {details['environment_desc']}")
                print(f"  Training Date: {details['date']}")
            
            # Offer to rename
            utility.rename_model_with_artifacts(
                current_path, 
                suggested_name, 
                dry_run=not args.execute
            )
            
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.generate_name:
        if not args.algorithm or not args.version:
            print("Error: --algorithm and --version are required for name generation")
            return
        
        name = utility.generate_model_name(
            algorithm=args.algorithm,
            version=args.version,
            environment=args.environment
        )
        
        print("Generated Model Name")
        print("=" * 30)
        print(f"Filename: {name}")
        
        details = utility.parse_model_name(name)
        if details.get('valid'):
            print(f"Algorithm: {details['algorithm_desc']}")
            print(f"Version: {details['version']}")
            print(f"Environment: {details['environment_desc']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
