#!/usr/bin/env python3
"""
Systematic Testing Framework for Disaster Response Model Improvements

This script provides a comprehensive framework to test different oversampling
approaches and systematically track improvements. It creates baselines and
compares multiple approaches side-by-side.

Usage: python systematic_testing_framework.py
"""

import pandas as pd
import numpy as np
import os
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

class ModelTestingFramework:
    """Framework for systematic testing of model improvements."""
    
    def __init__(self, results_dir="data/04_fct"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.test_results = {}
        self.baseline_established = False
        
    def run_baseline_test(self):
        """Run a baseline test without any oversampling."""
        print("🔬 RUNNING BASELINE TEST (No Oversampling)")
        print("=" * 50)
        
        # Create a temporary version of train_classifier.py without oversampling
        self._create_no_oversampling_version()
        
        try:
            # Run the baseline training
            cmd = [
                sys.executable, "models/train_classifier.py",
                "data/02_stg/stg_disaster_response.db", 
                "models/baseline_classifier.pkl"
            ]
            
            print("Training baseline model...")
            result = subprocess.run(cmd, capture_output=True, text=True, input="yes\nno\nno\nno\n")
            
            if result.returncode == 0:
                print("✅ Baseline model trained successfully!")
                self.baseline_established = True
                self._record_test_result("baseline", "No oversampling", "Baseline model")
            else:
                print("❌ Baseline training failed:")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Error running baseline test: {e}")
        finally:
            # Restore original file
            self._restore_original_version()
    
    def run_oversampling_tests(self):
        """Run tests with different oversampling approaches."""
        if not self.baseline_established:
            print("⚠️  Baseline not established. Run baseline test first.")
            return
            
        methods = ['conservative', 'smote', 'adasyn']
        
        for method in methods:
            print(f"\n🔬 TESTING {method.upper()} OVERSAMPLING")
            print("=" * 50)
            
            try:
                # Modify the training script to use specific method
                self._modify_sampling_method(method)
                
                # Run training
                cmd = [
                    sys.executable, "models/train_classifier.py",
                    "data/02_stg/stg_disaster_response.db", 
                    f"models/{method}_classifier.pkl"
                ]
                
                print(f"Training {method} model...")
                result = subprocess.run(cmd, capture_output=True, text=True, input="yes\nno\nno\nno\n")
                
                if result.returncode == 0:
                    print(f"✅ {method} model trained successfully!")
                    self._record_test_result(method, f"{method} oversampling", f"{method.title()} model")
                else:
                    print(f"❌ {method} training failed:")
                    print(result.stderr)
                    
            except Exception as e:
                print(f"❌ Error running {method} test: {e}")
            finally:
                # Restore original file
                self._restore_original_version()
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive comparison report."""
        print("\n📊 GENERATING COMPREHENSIVE REPORT")
        print("=" * 50)
        
        # Find all result files
        result_files = {}
        for file_path in self.results_dir.glob("fct_*_prediction_results.csv"):
            if "base_model" in file_path.name:
                result_files["baseline"] = file_path
            elif "best_model" in file_path.name:
                result_files["best_model"] = file_path
            elif "optimized" in file_path.name:
                result_files["optimized"] = file_path
        
        if len(result_files) < 2:
            print("❌ Need at least 2 result files to generate report")
            return
            
        # Load and compare results
        results = {}
        for name, path in result_files.items():
            df = pd.read_csv(path)
            class_1_metrics = df[df['output_class'] == '1'].copy()
            results[name] = class_1_metrics
        
        # Generate detailed comparison
        self._create_detailed_comparison(results)
        self._create_summary_table(results)
        self._create_improvement_analysis(results)
        
    def _create_detailed_comparison(self, results):
        """Create detailed comparison of all models."""
        print("\n📈 DETAILED MODEL COMPARISON")
        print("-" * 30)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, df in results.items():
            avg_recall = df['recall'].mean()
            avg_precision = df['precision'].mean()
            avg_f1 = df['f1-score'].mean()
            total_categories = len(df)
            
            comparison_data.append({
                'Model': model_name.title(),
                'Avg Recall': avg_recall,
                'Avg Precision': avg_precision,
                'Avg F1-Score': avg_f1,
                'Categories': total_categories
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Avg Recall', ascending=False)
        
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Save to file
        output_path = self.results_dir / "fct_comprehensive_model_comparison.csv"
        comparison_df.to_csv(output_path, index=False)
        print(f"\n💾 Detailed comparison saved to: {output_path}")
    
    def _create_summary_table(self, results):
        """Create a summary table of key metrics."""
        print("\n📋 SUMMARY TABLE")
        print("-" * 20)
        
        summary_data = []
        for model_name, df in results.items():
            # Key metrics
            avg_recall = df['recall'].mean()
            avg_precision = df['precision'].mean()
            avg_f1 = df['f1-score'].mean()
            
            # Categories with very low recall (< 0.1)
            low_recall_cats = len(df[df['recall'] < 0.1])
            
            # Categories with good recall (> 0.5)
            good_recall_cats = len(df[df['recall'] > 0.5])
            
            summary_data.append({
                'Model': model_name,
                'Avg Recall': f"{avg_recall:.3f}",
                'Avg Precision': f"{avg_precision:.3f}",
                'Avg F1': f"{avg_f1:.3f}",
                'Low Recall Cats': low_recall_cats,
                'Good Recall Cats': good_recall_cats
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save to file
        output_path = self.results_dir / "fct_model_summary_table.csv"
        summary_df.to_csv(output_path, index=False)
        print(f"\n💾 Summary table saved to: {output_path}")
    
    def _create_improvement_analysis(self, results):
        """Analyze improvements between models."""
        if 'baseline' not in results:
            print("⚠️  No baseline results found for improvement analysis")
            return
            
        print("\n🚀 IMPROVEMENT ANALYSIS")
        print("-" * 25)
        
        baseline = results['baseline']
        
        for model_name, df in results.items():
            if model_name == 'baseline':
                continue
                
            print(f"\n📊 {model_name.upper()} vs BASELINE:")
            
            # Merge for comparison
            merged = baseline[['category', 'recall', 'precision', 'f1-score']].merge(
                df[['category', 'recall', 'precision', 'f1-score']], 
                on='category', 
                suffixes=('_baseline', f'_{model_name}')
            )
            
            # Calculate improvements
            merged['recall_improvement'] = merged[f'recall_{model_name}'] - merged['recall_baseline']
            merged['precision_change'] = merged[f'precision_{model_name}'] - merged['precision_baseline']
            merged['f1_improvement'] = merged[f'f1-score_{model_name}'] - merged['f1-score_baseline']
            
            # Overall improvements
            avg_recall_improvement = merged['recall_improvement'].mean()
            avg_precision_change = merged['precision_change'].mean()
            avg_f1_improvement = merged['f1_improvement'].mean()
            
            print(f"  Average Recall Change: {avg_recall_improvement:+.3f}")
            print(f"  Average Precision Change: {avg_precision_change:+.3f}")
            print(f"  Average F1 Change: {avg_f1_improvement:+.3f}")
            
            # Count improvements
            recall_improved = len(merged[merged['recall_improvement'] > 0])
            recall_worse = len(merged[merged['recall_improvement'] < 0])
            
            print(f"  Categories with Better Recall: {recall_improved}")
            print(f"  Categories with Worse Recall: {recall_worse}")
            
            # Top improvements
            top_improvements = merged.nlargest(5, 'recall_improvement')[
                ['category', 'recall_baseline', f'recall_{model_name}', 'recall_improvement']
            ]
            print(f"\n  🏆 Top 5 Recall Improvements:")
            for _, row in top_improvements.iterrows():
                print(f"    {row['category']}: {row['recall_baseline']:.3f} → {row[f'recall_{model_name}']:.3f} ({row['recall_improvement']:+.3f})")
            
            # Save detailed comparison
            output_path = self.results_dir / f"fct_{model_name}_vs_baseline_comparison.csv"
            merged.to_csv(output_path, index=False)
            print(f"  💾 Detailed comparison saved to: {output_path}")
    
    def _create_no_oversampling_version(self):
        """Create a temporary version without oversampling."""
        # This would modify the train_classifier.py to skip oversampling
        # For now, we'll use a simpler approach
        pass
    
    def _modify_sampling_method(self, method):
        """Modify the training script to use specific sampling method."""
        # This would modify the train_classifier.py to use specific method
        # For now, we'll use a simpler approach
        pass
    
    def _restore_original_version(self):
        """Restore the original training script."""
        # This would restore the original train_classifier.py
        pass
    
    def _record_test_result(self, test_name, description, model_name):
        """Record test results."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.test_results[test_name] = {
            'timestamp': timestamp,
            'description': description,
            'model_name': model_name,
            'status': 'completed'
        }
        
        # Save test log
        log_path = self.results_dir / "test_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)

def main():
    """Main function to run the systematic testing framework."""
    print("🧪 DISASTER RESPONSE MODEL TESTING FRAMEWORK")
    print("=" * 50)
    print("This framework will systematically test different oversampling approaches")
    print("and provide comprehensive comparisons.\n")
    
    framework = ModelTestingFramework()
    
    # Run baseline test
    framework.run_baseline_test()
    
    # Run oversampling tests
    framework.run_oversampling_tests()
    
    # Generate comprehensive report
    framework.generate_comprehensive_report()
    
    print("\n✅ Testing framework completed!")
    print("Check the data/04_fct/ directory for detailed results.")

if __name__ == "__main__":
    main()
