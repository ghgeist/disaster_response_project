#!/usr/bin/env python3
"""
Automated experiment runner for disaster response classification.
Runs all sampling experiments sequentially and generates a comprehensive report.
"""

import subprocess
import sys
import time
import os
import shlex
from datetime import datetime
import json

def run_experiment(experiment_name, sampling_method, model_name):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"🧪 RUNNING EXPERIMENT: {experiment_name}")
    print(f"📊 Sampling Method: {sampling_method}")
    print(f"💾 Model: {model_name}")
    print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Basic input validation to prevent command injection
        if not experiment_name or not isinstance(experiment_name, str):
            raise ValueError("Invalid experiment_name")
        if not sampling_method or not isinstance(sampling_method, str):
            raise ValueError("Invalid sampling_method")
        if not model_name or not isinstance(model_name, str):
            raise ValueError("Invalid model_name")
        
        # Sanitize model_name to prevent path traversal
        model_name = os.path.basename(model_name)  # Remove any path components
        if not model_name.endswith('.pkl'):
            model_name += '.pkl'
        
        # Basic character validation for model name (check only the base name, not extension)
        base_name = model_name[:-4] if model_name.endswith('.pkl') else model_name
        if not base_name or not all(c.isalnum() or c in '._-' for c in base_name):
            raise ValueError(f"Invalid characters in model_name: {model_name}")
        
        # Determine the input based on experiment type
        if sampling_method == "baseline":
            input_data = "1\n"  # baseline_no_sampling
        elif sampling_method == "smote":
            input_data = "2\n"  # smote_conservative
        elif sampling_method == "adasyn":
            input_data = "3\n"  # adasyn_moderate
        elif sampling_method == "conservative":
            input_data = "4\n"  # conservative_sampling
        else:
            input_data = "1\n"  # default to baseline
        
        # Run the training script with validated inputs
        cmd = [
            sys.executable, 
            "scripts/train_experiment.py",
            "data/02_stg/stg_disaster_response.db",
            f"models/{model_name}"
        ]
        
        result = subprocess.run(
            cmd, 
            input=input_data, 
            text=True, 
            capture_output=True,
            timeout=1800  # 30 minute timeout per experiment
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {experiment_name} completed successfully!")
            print(f"⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            return {
                'experiment': experiment_name,
                'status': 'success',
                'duration': duration,
                'output': result.stdout,
                'error': None
            }
        else:
            print(f"❌ {experiment_name} failed!")
            print(f"Error: {result.stderr}")
            return {
                'experiment': experiment_name,
                'status': 'failed',
                'duration': duration,
                'output': result.stdout,
                'error': result.stderr
            }
            

    except subprocess.TimeoutExpired:
        print(f"⏰ {experiment_name} timed out after 30 minutes!")
        return {
            'experiment': experiment_name,
            'status': 'timeout',
            'duration': 1800,
            'output': '',
            'error': 'Timeout after 30 minutes'
        }
    except Exception as e:
        print(f"💥 {experiment_name} crashed: {e}")
        return {
            'experiment': experiment_name,
            'status': 'crashed',
            'duration': 0,
            'output': '',
            'error': str(e)
        }

def generate_summary_report(results):
    """Generate a summary report of all experiments."""
    print(f"\n{'='*80}")
    print("📋 EXPERIMENT SUMMARY REPORT")
    print(f"{'='*80}")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print(f"✅ Successful experiments: {len(successful)}")
    print(f"❌ Failed experiments: {len(failed)}")
    
    if successful:
        print(f"\n📊 SUCCESSFUL EXPERIMENTS:")
        for result in successful:
            print(f"  • {result['experiment']}: {result['duration']/60:.1f} minutes")
    
    if failed:
        print(f"\n💥 FAILED EXPERIMENTS:")
        for result in failed:
            print(f"  • {result['experiment']}: {result['status']} - {result['error']}")
    
    total_time = sum(r['duration'] for r in results)
    print(f"\n⏱️  Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    
    # Save detailed results
    report_file = f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'total_duration': total_time,
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {report_file}")

def main():
    """Run all sampling experiments."""
    print("🚀 AUTOMATED DISASTER RESPONSE EXPERIMENT RUNNER")
    print("=" * 60)
    print("This script will run all sampling experiments automatically.")
    print("Perfect for running while you work out! 💪")
    print("=" * 60)
    
    # Define experiments to run
    experiments = [
        ("baseline_no_sampling", "baseline", "baseline_model"),
        ("smote_conservative", "smote", "smote_model"),
        ("adasyn_moderate", "adasyn", "adasyn_model"),
        ("conservative_sampling", "conservative", "conservative_model")
    ]
    
    print(f"\n📋 Will run {len(experiments)} experiments:")
    for i, (name, method, model) in enumerate(experiments, 1):
        print(f"  {i}. {name} ({method})")
    
    print(f"\n⏰ Estimated total time: {len(experiments) * 5} minutes")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all experiments
    results = []
    for experiment_name, sampling_method, model_name in experiments:
        result = run_experiment(experiment_name, sampling_method, model_name)
        results.append(result)
        
        # Small break between experiments
        if result['status'] == 'success':
            print("⏸️  Taking a 10-second break before next experiment...")
            time.sleep(10)
    
    # Generate summary report
    generate_summary_report(results)
    
    print(f"\n🎉 All experiments completed!")
    print(f"🕐 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 Check these directories for results:")
    print(f"  • experiments/ - Individual experiment results")
    print(f"  • models/ - Trained model files")
    print(f"  • data/04_fct/ - Performance metrics")
    print(f"\n💡 Next steps:")
    print(f"  • Run: python scripts/compare_models.py")
    print(f"  • Review results in experiments/ directories")
    print(f"  • Test best model in web application")

if __name__ == "__main__":
    main()
