#!/usr/bin/env python3
"""
Enhanced model comparison tool for disaster response classification CSV results.

This script provides crystal-clear comparison between different model prediction results
stored in CSV format, making it easy for portfolio reviewers to understand model performance improvements.
"""

# Standard library imports
import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

# Third-party imports
import numpy as np
import pandas as pd


def load_csv_results(file_path: str) -> pd.DataFrame:
    """
    Load prediction results from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with prediction results
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Validate expected columns
    expected_columns = ['category', 'output_class', 'precision', 'recall', 'f1-score', 'support']
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")
    
    return df


def calculate_metrics_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary metrics from prediction results DataFrame.
    
    Args:
        df: DataFrame with prediction results
        
    Returns:
        Dictionary with summary metrics
    """
    # Filter for weighted averages only (most representative)
    weighted_avg = df[df['output_class'] == 'weighted avg'].copy()
    
    if weighted_avg.empty:
        return {}
    
    # Calculate overall metrics
    metrics = {
        'precision_mean': weighted_avg['precision'].mean(),
        'recall_mean': weighted_avg['recall'].mean(),
        'f1_score_mean': weighted_avg['f1-score'].mean(),
        'precision_std': weighted_avg['precision'].std(),
        'recall_std': weighted_avg['recall'].std(),
        'f1_score_std': weighted_avg['f1-score'].std(),
        'num_categories': len(weighted_avg),
        'categories': weighted_avg['category'].tolist()
    }
    
    return metrics


def compare_csv_models(file_paths: List[str], model_names: List[str] = None) -> pd.DataFrame:
    """
    Compare multiple CSV model results and return a summary DataFrame.
    
    Args:
        file_paths: List of paths to CSV files
        model_names: Optional list of model names (defaults to file names)
        
    Returns:
        DataFrame with comparison results
    """
    if model_names is None:
        model_names = [os.path.basename(f).replace('.csv', '') for f in file_paths]
    
    if len(file_paths) != len(model_names):
        raise ValueError("Number of file paths must match number of model names")
    
    comparison_data = []
    
    for file_path, model_name in zip(file_paths, model_names):
        try:
            df = load_csv_results(file_path)
            metrics = calculate_metrics_summary(df)
            
            comparison_data.append({
                'model': model_name,
                'file_path': file_path,
                'precision_mean': metrics.get('precision_mean', 0),
                'recall_mean': metrics.get('recall_mean', 0),
                'f1_score_mean': metrics.get('f1_score_mean', 0),
                'precision_std': metrics.get('precision_std', 0),
                'recall_std': metrics.get('recall_std', 0),
                'f1_score_std': metrics.get('f1_score_std', 0),
                'num_categories': metrics.get('num_categories', 0),
                'status': 'loaded'
            })
        except Exception as e:
            comparison_data.append({
                'model': model_name,
                'file_path': file_path,
                'precision_mean': 0,
                'recall_mean': 0,
                'f1_score_mean': 0,
                'precision_std': 0,
                'recall_std': 0,
                'f1_score_std': 0,
                'num_categories': 0,
                'status': f'error: {str(e)}'
            })
    
    return pd.DataFrame(comparison_data)


def detailed_category_comparison(file_paths: List[str], model_names: List[str] = None) -> pd.DataFrame:
    """
    Create detailed category-by-category comparison.
    
    Args:
        file_paths: List of paths to CSV files
        model_names: Optional list of model names
        
    Returns:
        DataFrame with detailed comparison by category
    """
    if model_names is None:
        model_names = [os.path.basename(f).replace('.csv', '') for f in file_paths]
    
    all_data = []
    
    for file_path, model_name in zip(file_paths, model_names):
        try:
            df = load_csv_results(file_path)
            # Filter for weighted averages
            weighted_avg = df[df['output_class'] == 'weighted avg'].copy()
            weighted_avg['model'] = model_name
            all_data.append(weighted_avg)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Pivot for comparison
    comparison_df = combined_df.pivot_table(
        index='category',
        columns='model',
        values=['precision', 'recall', 'f1-score'],
        aggfunc='first'
    )
    
    return comparison_df


def calculate_improvements(base_df: pd.DataFrame, production_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance improvements between base and production models.
    
    Args:
        base_df: Base model results DataFrame
        production_df: Production model results DataFrame
        
    Returns:
        DataFrame with improvement metrics
    """
    # Filter for weighted averages
    base_weighted = base_df[base_df['output_class'] == 'weighted avg'].copy()
    prod_weighted = production_df[production_df['output_class'] == 'weighted avg'].copy()
    
    # Merge on category
    merged = base_weighted.merge(
        prod_weighted, 
        on='category', 
        suffixes=('_base', '_production')
    )
    
    # Calculate improvements
    improvements = []
    for _, row in merged.iterrows():
        category = row['category']
        
        # Calculate percentage changes
        precision_change = ((row['precision_production'] - row['precision_base']) / row['precision_base']) * 100
        recall_change = ((row['recall_production'] - row['recall_base']) / row['recall_base']) * 100
        f1_change = ((row['f1-score_production'] - row['f1-score_base']) / row['f1-score_base']) * 100
        
        improvements.append({
            'category': category,
            'precision_base': row['precision_base'],
            'precision_production': row['precision_production'],
            'precision_change_pct': precision_change,
            'recall_base': row['recall_base'],
            'recall_production': row['recall_production'],
            'recall_change_pct': recall_change,
            'f1_base': row['f1-score_base'],
            'f1_production': row['f1-score_production'],
            'f1_change_pct': f1_change,
            'support': row['support_base']  # Support should be the same
        })
    
    return pd.DataFrame(improvements)


def print_model_summary(model_name: str, file_path: str):
    """
    Print a detailed summary of a single model.
    
    Args:
        model_name: Name of the model
        file_path: Path to the CSV file
    """
    try:
        df = load_csv_results(file_path)
        metrics = calculate_metrics_summary(df)
        
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"FILE: {file_path}")
        print(f"{'='*60}")
        
        print(f"\n📊 Overall Performance (Weighted Average):")
        print(f"   Precision: {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}")
        print(f"   Recall:    {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score_mean']:.4f} ± {metrics['f1_score_std']:.4f}")
        print(f"   Categories: {metrics['num_categories']}")
        
        # Show top and bottom performing categories
        weighted_avg = df[df['output_class'] == 'weighted avg'].copy()
        weighted_avg_sorted = weighted_avg.sort_values('f1-score', ascending=False)
        
        print(f"\n🏆 Top 5 Categories by F1-Score:")
        for _, row in weighted_avg_sorted.head().iterrows():
            print(f"   {row['category']:<20} F1: {row['f1-score']:.4f}")
        
        print(f"\n📉 Bottom 5 Categories by F1-Score:")
        for _, row in weighted_avg_sorted.tail().iterrows():
            print(f"   {row['category']:<20} F1: {row['f1-score']:.4f}")
            
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")


def main():
    """
    Main function with command-line interface for CSV model comparison.
    """
    parser = argparse.ArgumentParser(description='Compare disaster response classification models from CSV files')
    parser.add_argument('files', nargs='+', help='CSV files to compare')
    parser.add_argument('--names', nargs='+', help='Model names (optional)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed category comparison')
    parser.add_argument('--improvements', action='store_true', help='Show performance improvements (requires exactly 2 files)')
    
    args = parser.parse_args()
    
    print("🔬 Disaster Response Classification - CSV Model Comparison Tool")
    print("=" * 70)
    
    try:
        # Basic comparison
        comparison_df = compare_csv_models(args.files, args.names)
        
        print(f"\n📊 Model Comparison Summary:")
        print("-" * 70)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Detailed category comparison
        if args.detailed:
            print(f"\n📋 Detailed Category Comparison:")
            print("-" * 70)
            detailed_df = detailed_category_comparison(args.files, args.names)
            if not detailed_df.empty:
                print(detailed_df.to_string(float_format='%.4f'))
        
        # Performance improvements (if exactly 2 files)
        if args.improvements and len(args.files) == 2:
            print(f"\n📈 Performance Improvements (Production vs Base):")
            print("-" * 70)
            
            base_df = load_csv_results(args.files[0])
            prod_df = load_csv_results(args.files[1])
            improvements_df = calculate_improvements(base_df, prod_df)
            
            if not improvements_df.empty:
                # Show summary statistics
                avg_improvements = {
                    'precision': improvements_df['precision_change_pct'].mean(),
                    'recall': improvements_df['recall_change_pct'].mean(),
                    'f1_score': improvements_df['f1_change_pct'].mean()
                }
                
                print(f"Average Improvements:")
                print(f"   Precision: {avg_improvements['precision']:+.2f}%")
                print(f"   Recall:    {avg_improvements['recall']:+.2f}%")
                print(f"   F1-Score:  {avg_improvements['f1_score']:+.2f}%")
                
                print(f"\nCategory-by-Category Improvements:")
                print(improvements_df[['category', 'f1_change_pct', 'precision_change_pct', 'recall_change_pct']].to_string(index=False, float_format='%.2f'))
        
        # Individual model summaries
        print(f"\n📋 Individual Model Details:")
        for i, (file_path, model_name) in enumerate(zip(args.files, args.names or [os.path.basename(f).replace('.csv', '') for f in args.files])):
            print_model_summary(model_name, file_path)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
