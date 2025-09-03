#!/usr/bin/env python3
"""
Run a set of sampling experiments by importing the training function directly.

This avoids interactive stdin and subprocess overhead. Adjust the EXPERIMENTS
list or add CLI args later if needed.
"""

import os
import sys
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.train_model import train_experiment, create_experiment_name  # type: ignore


DB_PATH = 'data/02_stg/stg_disaster_response.db'
EXPERIMENTS = [
    ('baseline', 'baseline'),
    ('smote', 'smote'),
    ('adasyn', 'adasyn'),
    ('conservative', 'conservative'),
]


def main():
    print('🚀 Running experiments (import-based)')
    for label, method in EXPERIMENTS:
        name = create_experiment_name(method)
        print(f"\n=== {label.upper()} ({method}) ===")
        train_experiment(name, method, DB_PATH)
        time.sleep(3)


if __name__ == '__main__':
    main()


