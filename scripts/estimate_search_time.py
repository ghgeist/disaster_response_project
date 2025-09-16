#!/usr/bin/env python3
"""
Estimates the runtime of a hyperparameter search without running the full search.
"""

import argparse
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Add src to path to allow for imports from the disasterproject package
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from disasterproject.data.loader import load_data
from disasterproject.data.preprocessor import tokenize
from disasterproject.models.pipeline import run_parameter_search
from disasterproject.utils.json_io import load_hyperparameter_optimization_config

def create_pipeline():
    """Create the text classification Pipeline compatible with our parameter grids."""
    return Pipeline([
        (
            "vect",
            CountVectorizer(analyzer=tokenize, token_pattern=None, lowercase=False),
        ),
        ("tfidf", TfidfTransformer(smooth_idf=False)),
        (
            "clf",
            MultiOutputClassifier(
                RandomForestClassifier(n_jobs=1)
            ),
        ),
    ])

def main():
    """Main function to estimate search runtime."""
    parser = argparse.ArgumentParser(description="Estimate hyperparameter search runtime.")
    parser.add_argument("database_filepath", help="Path to the SQLite database file.")
    parser.add_argument("config_filepath", help="Path to the hyperparameter configuration JSON file.")
    args = parser.parse_args()

    print(f"Loading data from {args.database_filepath}...")
    X, Y = load_data(args.database_filepath)
    if X is None:
        sys.exit(1)

    print(f"Loading configuration from {args.config_filepath}...")
    parameters = load_hyperparameter_optimization_config(args.config_filepath)
    if parameters is None:
        sys.exit(1)

    print("Creating pipeline...")
    pipeline = create_pipeline()

    print("Estimating runtime...")
    run_parameter_search(pipeline, parameters, X, Y, use_small_subset=True)

if __name__ == "__main__":
    main()
