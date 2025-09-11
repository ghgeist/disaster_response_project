#!/usr/bin/env python3
"""
Batch experiment runner for disaster response classification.

This script runs multiple sampling experiments sequentially without user interaction.
Use this for automated experiment runs when you want to compare different sampling strategies.

Usage:
    python scripts/run_batch_experiments.py

The script will run all predefined experiments:
- baseline (no sampling)
- smote (SMOTE conservative sampling)  
- adasyn (ADASYN moderate sampling)
- conservative (very conservative SMOTE sampling)

Results are saved to the experiments/ directory structure.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.run_experiment import train_experiment
from disasterproject.utils.experiment_tracker import create_experiment_name
from disasterproject.utils.config import setup_logging


DB_PATH = 'data/02_stg/stg_disaster_response.db'
EXPERIMENTS = [
    ('baseline', None, 'No sampling - establish baseline'),
    ('class_weights', 'weights', 'Use class weights instead of resampling'),
    ('mlsmote_conservative', 'mlsmote', 'ML-SMOTE with k=3, ratio=0.3'),
    ('random_oversample', 'random', 'Random oversampling to 50% majority'),
]


def main():
    """Run all predefined experiments in batch mode."""
    # Set up logging
    setup_logging()
    
    print('🚀 Batch Experiment Runner')
    print('=' * 50)
    print(f'Database: {DB_PATH}')
    print(f'Experiments to run: {len(EXPERIMENTS)}')
    print(f'Started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 50)
    
    start_time = time.time()
    results = []
    
    for i, (label, method, description) in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] Starting: {label.upper()}")
        print(f"Description: {description}")
        print(f"Method: {method}")
        print('-' * 40)
        
        experiment_start = time.time()
        
        try:
            name = create_experiment_name(method)
            model = train_experiment(name, method, DB_PATH)
            
            experiment_duration = time.time() - experiment_start
            
            if model is not None:
                print(f"✅ {label} completed successfully in {experiment_duration/60:.1f} minutes")
                results.append({
                    'experiment': label,
                    'method': method,
                    'status': 'success',
                    'duration': experiment_duration
                })
            else:
                print(f"❌ {label} failed")
                results.append({
                    'experiment': label,
                    'method': method,
                    'status': 'failed',
                    'duration': experiment_duration
                })
                
        except Exception as e:
            experiment_duration = time.time() - experiment_start
            print(f"💥 {label} crashed: {e}")
            logging.error(f"Experiment {label} failed with error: {e}")
            results.append({
                'experiment': label,
                'method': method,
                'status': 'crashed',
                'duration': experiment_duration,
                'error': str(e)
            })
        
        # Brief pause between experiments
        if i < len(EXPERIMENTS):
            print("⏸️  Pausing 3 seconds before next experiment...")
            time.sleep(3)
    
    # Summary report
    total_duration = time.time() - start_time
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print('\n' + '=' * 60)
    print('📋 BATCH EXPERIMENT SUMMARY')
    print('=' * 60)
    print(f'✅ Successful: {len(successful)}/{len(EXPERIMENTS)}')
    print(f'❌ Failed: {len(failed)}/{len(EXPERIMENTS)}')
    print(f'⏱️  Total time: {total_duration/60:.1f} minutes')
    print(f'🕐 Finished at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    if successful:
        print('\n📊 Successful experiments:')
        for result in successful:
            print(f"  • {result['experiment']}: {result['duration']/60:.1f} min")
    
    if failed:
        print('\n💥 Failed experiments:')
        for result in failed:
            error_msg = result.get('error', 'Unknown error')
            print(f"  • {result['experiment']}: {result['status']} - {error_msg}")
    
    print('\n📁 Check these directories for results:')
    print('  • experiments/configs/ - Experiment configurations')
    print('  • experiments/results/ - Result summaries')  
    print('  • experiments/models/ - Trained models')
    print('  • data/04_fct/ - Performance metrics')
    
    print('\n💡 Next steps:')
    print('  • Run: python scripts/compare_models.py')
    print('  • Review results in experiments/ directories')


if __name__ == '__main__':
    main()
