#!/usr/bin/env python3
"""
Quick script to compare model results and see if SMOTE helped.
Usage: python compare_results.py
"""

import pandas as pd
import os

def compare_model_results():
    """Compare base model vs best model results to see SMOTE impact."""
    
    # File paths - using os.path.join for platform agnostic paths
    base_results = os.path.join("data", "04_fct", "fct_base_model_prediction_results.csv")
    best_results = os.path.join("data", "04_fct", "fct_best_model_prediction_results.csv")
    old_results = os.path.join("data", "04_fct", "fct_prediction_results.csv")  # Your original results
    
    # Check which files exist
    files_to_compare = []
    if os.path.exists(old_results):
        files_to_compare.append(("Original", old_results))
    if os.path.exists(base_results):
        files_to_compare.append(("SMOTE Base", base_results))
    if os.path.exists(best_results):
        files_to_compare.append(("Best Model", best_results))
    
    if len(files_to_compare) < 2:
        print("Need at least 2 result files to compare!")
        print("Available files:")
        for name, path in files_to_compare:
            print(f"  - {name}: {path}")
        return
    
    # Load and process results
    results = {}
    for name, path in files_to_compare:
        df = pd.read_csv(path)
        # Focus on class "1" (positive class) metrics
        class_1_metrics = df[df['output_class'] == '1'].copy()
        results[name] = class_1_metrics
    
    # Get categories with worst recall in original model
    if "Original" in results:
        worst_recall = results["Original"].nsmallest(10, 'recall')[['category', 'recall', 'precision', 'f1-score']]
        print("=== 10 WORST RECALL CATEGORIES IN ORIGINAL MODEL ===")
        print(worst_recall.to_string(index=False, float_format='%.3f'))
        print()
    
    # Compare recall improvements
    if len(results) >= 2:
        models = list(results.keys())
        model_a, model_b = models[0], models[-1]  # First vs last
        
        # Merge on category to compare
        merged = results[model_a][['category', 'recall', 'precision', 'f1-score']].merge(
            results[model_b][['category', 'recall', 'precision', 'f1-score']], 
            on='category', 
            suffixes=(f'_{model_a}', f'_{model_b}')
        )
        
        # Calculate improvements
        merged['recall_improvement'] = merged[f'recall_{model_b}'] - merged[f'recall_{model_a}']
        merged['precision_change'] = merged[f'precision_{model_b}'] - merged[f'precision_{model_a}']
        
        # Show biggest improvements
        print(f"=== BIGGEST RECALL IMPROVEMENTS: {model_a} → {model_b} ===")
        improvements = merged.nlargest(10, 'recall_improvement')[
            ['category', f'recall_{model_a}', f'recall_{model_b}', 'recall_improvement']
        ]
        print(improvements.to_string(index=False, float_format='%.3f'))
        print()
        
        # Show categories that got worse
        worse = merged[merged['recall_improvement'] < -0.05]  # More than 5% worse
        if len(worse) > 0:
            print(f"=== CATEGORIES THAT GOT SIGNIFICANTLY WORSE ===")
            worse_display = worse[['category', f'recall_{model_a}', f'recall_{model_b}', 'recall_improvement']]
            print(worse_display.to_string(index=False, float_format='%.3f'))
            print()
        
        # Overall summary
        avg_recall_before = merged[f'recall_{model_a}'].mean()
        avg_recall_after = merged[f'recall_{model_b}'].mean()
        print(f"=== OVERALL SUMMARY ===")
        print(f"Average Recall - {model_a}: {avg_recall_before:.3f}")
        print(f"Average Recall - {model_b}: {avg_recall_after:.3f}")
        print(f"Overall Change: {avg_recall_after - avg_recall_before:+.3f}")
        print(f"Categories Improved: {len(merged[merged['recall_improvement'] > 0])}")
        print(f"Categories Worse: {len(merged[merged['recall_improvement'] < 0])}")

if __name__ == "__main__":
    compare_model_results()