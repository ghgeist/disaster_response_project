#!/usr/bin/env python3
"""
Analyze vocabulary distribution from a trained model.

Extracts vocabulary, document frequencies, and provides recommendations
for optimal max_features, min_df, and max_df values.
"""

# Standard library imports
import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

# Third-party imports
import joblib
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Local imports
from disasterproject.utils.config import CRITICAL_LABELS, setup_logging

def analyze_vocabulary(model_path, output_path=None):
    """
    Analyze vocabulary from a trained model.
    
    Args:
        model_path: Path to trained model pickle file
        output_path: Optional path to save analysis JSON
    
    Returns:
        dict: Analysis results with recommendations
    """
    print(f"\n{'='*70}")
    print("VOCABULARY DISTRIBUTION ANALYSIS")
    print(f"{'='*70}\n")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    # Extract vectorizer
    if 'vect' not in model.named_steps:
        raise ValueError("Model pipeline does not contain 'vect' step")
    
    vectorizer = model.named_steps['vect']
    
    # Get vocabulary
    try:
        feature_names = vectorizer.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn
        feature_names = vectorizer.get_feature_names()
    
    vocab_size = len(feature_names)
    print(f"✓ Vocabulary size: {vocab_size:,} features")
    
    # Get document frequencies (if available)
    if hasattr(vectorizer, 'vocabulary_'):
        # CountVectorizer stores vocabulary as dict mapping term -> index
        vocab_dict = vectorizer.vocabulary_
        # Reverse mapping: index -> term
        index_to_term = {idx: term for term, idx in vocab_dict.items()}
    else:
        index_to_term = {i: name for i, name in enumerate(feature_names)}
    
    # Analyze n-gram distribution
    unigrams = []
    bigrams = []
    for term in feature_names:
        words = term.split()
        if len(words) == 1:
            unigrams.append(term)
        elif len(words) == 2:
            bigrams.append(term)
    
    print(f"  - Unigrams: {len(unigrams):,}")
    print(f"  - Bigrams: {len(bigrams):,}")
    
    # Check for critical disaster terms
    critical_terms = {
        'water', 'food', 'shelter', 'medical', 'help', 'search', 'rescue',
        'security', 'hospital', 'aid', 'emergency', 'urgent', 'need'
    }
    
    found_critical = []
    missing_critical = []
    
    for term in critical_terms:
        # Check if term appears in vocabulary (as unigram or part of bigram)
        found = False
        for feature in feature_names:
            if term in feature.lower():
                found = True
                found_critical.append((term, feature))
                break
        if not found:
            missing_critical.append(term)
    
    print(f"\n✓ Critical terms found: {len(found_critical)}/{len(critical_terms)}")
    if missing_critical:
        print(f"  ⚠ Missing: {', '.join(missing_critical)}")
    
    # Estimate document frequencies (approximate from IDF if available)
    # Note: We can't get exact DF without retraining, but we can estimate
    # from TF-IDF idf_ values if available
    
    # Get TF-IDF transformer
    tfidf = model.named_steps.get('tfidf')
    idf_values = None
    if tfidf is not None and hasattr(tfidf, 'idf_'):
        idf_values = tfidf.idf_
        print(f"\n✓ TF-IDF IDF values available")
    
    # Create vocabulary DataFrame
    vocab_df = pd.DataFrame({
        'term': feature_names,
        'is_unigram': [len(term.split()) == 1 for term in feature_names],
        'is_bigram': [len(term.split()) == 2 for term in feature_names]
    })
    
    if idf_values is not None:
        vocab_df['idf'] = idf_values
        # Estimate document frequency from IDF
        # IDF = log((N + 1) / (df + 1)) + 1 (with smoothing)
        # Solving for df: df ≈ (N + 1) / exp(IDF - 1) - 1
        # We don't know N exactly, but can estimate
        # For typical disaster response dataset: N ≈ 26,000
        N_estimate = 26000
        vocab_df['estimated_df'] = (N_estimate + 1) / np.exp(idf_values - 1) - 1
        vocab_df['estimated_df'] = vocab_df['estimated_df'].clip(lower=1)
        vocab_df['estimated_df_pct'] = (vocab_df['estimated_df'] / N_estimate * 100).clip(upper=100)
    
    # Sort by estimated frequency (if available) or alphabetically
    if 'estimated_df' in vocab_df.columns:
        vocab_df = vocab_df.sort_values('estimated_df', ascending=False)
    else:
        vocab_df = vocab_df.sort_values('term')
    
    # Analyze frequency distribution
    print(f"\n{'='*70}")
    print("FREQUENCY DISTRIBUTION ANALYSIS")
    print(f"{'='*70}\n")
    
    if 'estimated_df' in vocab_df.columns:
        print("Top 20 most frequent terms:")
        print(vocab_df.head(20)[['term', 'estimated_df', 'estimated_df_pct']].to_string(index=False))
        
        print("\nBottom 20 least frequent terms:")
        print(vocab_df.tail(20)[['term', 'estimated_df', 'estimated_df_pct']].to_string(index=False))
        
        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print("\nDocument Frequency Percentiles:")
        for p in percentiles:
            val = np.percentile(vocab_df['estimated_df'], p)
            pct = np.percentile(vocab_df['estimated_df_pct'], p)
            print(f"  {p:2d}th percentile: {val:6.1f} docs ({pct:5.2f}%)")
        
        # Recommendations
        print(f"\n{'='*70}")
        print("RECOMMENDATIONS")
        print(f"{'='*70}\n")
        
        # min_df recommendation
        df_2 = np.sum(vocab_df['estimated_df'] >= 2)
        df_3 = np.sum(vocab_df['estimated_df'] >= 3)
        print(f"Terms appearing in ≥2 documents: {df_2:,} ({df_2/vocab_size*100:.1f}%)")
        print(f"Terms appearing in ≥3 documents: {df_3:,} ({df_3/vocab_size*100:.1f}%)")
        print(f"  → Recommended min_df: 2 (removes {vocab_size-df_2:,} rare terms)")
        
        # max_df recommendation
        df_95 = np.sum(vocab_df['estimated_df_pct'] <= 95)
        df_90 = np.sum(vocab_df['estimated_df_pct'] <= 90)
        print(f"\nTerms appearing in ≤95% of documents: {df_95:,} ({df_95/vocab_size*100:.1f}%)")
        print(f"Terms appearing in ≤90% of documents: {df_90:,} ({df_90/vocab_size*100:.1f}%)")
        print(f"  → Recommended max_df: 0.95 (removes universal terms)")
        
        # max_features recommendations
        print(f"\nVocabulary size at different cutoffs:")
        for cutoff in [15000, 20000, 25000, 30000]:
            if cutoff < vocab_size:
                # Estimate how many features would be kept
                # Assuming we keep top N by frequency
                kept = min(cutoff, vocab_size)
                pct = kept / vocab_size * 100
                print(f"  max_features={cutoff:,}: {kept:,} features ({pct:.1f}% of current)")
        
    else:
        print("⚠ IDF values not available - cannot estimate document frequencies")
        print("  Recommendations will be based on vocabulary size only")
    
    # Check critical terms positions
    print(f"\n{'='*70}")
    print("CRITICAL TERM POSITIONS")
    print(f"{'='*70}\n")
    
    critical_positions = {}
    for term, feature in found_critical:
        if 'estimated_df' in vocab_df.columns:
            # Find position by frequency
            term_df = vocab_df[vocab_df['term'] == feature]
            if len(term_df) > 0:
                rank = vocab_df.index.get_loc(term_df.index[0]) + 1
                critical_positions[term] = {
                    'feature': feature,
                    'rank': rank,
                    'percentile': (vocab_size - rank) / vocab_size * 100
                }
    
    if critical_positions:
        print("Critical terms in vocabulary (by frequency rank):")
        for term, info in sorted(critical_positions.items(), key=lambda x: x[1]['rank']):
            print(f"  {term:15s}: rank {info['rank']:6,} ({info['percentile']:5.1f}th percentile) - '{info['feature']}'")
    
    # Prepare output
    analysis = {
        'vocabulary_size': int(vocab_size),
        'unigrams': int(len(unigrams)),
        'bigrams': int(len(bigrams)),
        'critical_terms_found': len(found_critical),
        'critical_terms_missing': missing_critical,
        'recommendations': {
            'min_df': 2,
            'max_df': 0.95,
            'max_features_options': [15000, 20000, 25000, 30000]
        }
    }
    
    if 'estimated_df' in vocab_df.columns:
        analysis['frequency_stats'] = {
            'mean_df': float(vocab_df['estimated_df'].mean()),
            'median_df': float(vocab_df['estimated_df'].median()),
            'min_df': float(vocab_df['estimated_df'].min()),
            'max_df': float(vocab_df['estimated_df'].max()),
            'terms_with_df_ge_2': int(df_2),
            'terms_with_df_ge_3': int(df_3),
            'terms_with_df_pct_le_95': int(df_95),
            'terms_with_df_pct_le_90': int(df_90)
        }
        
        analysis['percentiles'] = {
            f'p{p}': float(np.percentile(vocab_df['estimated_df'], p))
            for p in [10, 25, 50, 75, 90, 95, 99]
        }
    
    if critical_positions:
        analysis['critical_term_ranks'] = {
            term: {'rank': info['rank'], 'percentile': info['percentile']}
            for term, info in critical_positions.items()
        }
    
    # Save output
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n✓ Analysis saved to: {output_path}")
    
    print(f"\n{'='*70}\n")
    
    return analysis

def main():
    parser = argparse.ArgumentParser(
        description='Analyze vocabulary distribution from trained model'
    )
    parser.add_argument(
        '--model-path',
        default='experiments/experimental_runs/2025-11-04/lr_baseline_model.pkl',
        help='Path to trained model pickle file'
    )
    parser.add_argument(
        '--output',
        default='experiments/experimental_runs/2025-11-04/vocabulary_analysis.json',
        help='Path to save analysis JSON output'
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        analysis = analyze_vocabulary(args.model_path, args.output)
        print("✓ Vocabulary analysis completed successfully")
    except Exception as e:
        logging.error(f"Vocabulary analysis failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

