#!/usr/bin/env python3
"""
Create a frozen evaluation set (by stable UID) from the SQLite DB.

This script writes a JSON file of UIDs that define the eval (holdout) split so that
all future models can be compared on the exact same examples.

UIDs are computed as SHA-1 of "<message>|<row_index>", where row_index comes
from the order returned by the loader. This keeps IDs stable as long as the
underlying dataset ordering and content do not change.

Usage:
    python scripts/create_frozen_eval_ids.py \
        --db data/02_stg/stg_disaster_response.db \
        --out experiments/experimental_configs/eval_sets/eval_ids.json \
        --test-size 0.2 --seed 42
"""

# Standard library imports
import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime

# Third-party imports
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Local imports
from disasterproject.utils.config import setup_logging
from disasterproject.data.loader import load_data


def compute_uids(messages):
    """
    Compute stable UIDs from messages and their positional index.

    Args:
        messages (Iterable[str]): Ordered collection of message texts

    Returns:
        List[str]: Hex SHA-1 UIDs
    """
    uids = []
    for idx, msg in enumerate(messages):
        text = '' if msg is None else str(msg)
        uid_src = f"{text}|{idx}"
        uids.append(hashlib.sha1(uid_src.encode('utf-8')).hexdigest())
    return uids


def main():
    parser = argparse.ArgumentParser(
        description='Create a frozen eval set CSV of UIDs for reproducible evaluation.'
    )
    parser.add_argument('--db', dest='database_filepath',
                        default='data/02_stg/stg_disaster_response.db',
                        help='Path to SQLite database (default: data/02_stg/stg_disaster_response.db)')
    parser.add_argument('--out', dest='out_json',
                        default='experiments/experimental_configs/eval_sets/eval_ids.json',
                        help='Output JSON path for eval UIDs (default: experiments/experimental_configs/eval_sets/eval_ids.json)')
    parser.add_argument('--test-size', dest='test_size', type=float, default=0.2,
                        help='Holdout fraction (default: 0.2)')
    parser.add_argument('--seed', dest='seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    setup_logging()
    logging.info('Loading data from %s', args.database_filepath)
    X, Y = load_data(args.database_filepath)
    if X is None or Y is None:
        logging.error('Failed to load data. Exiting.')
        sys.exit(1)

    logging.info('Computing UIDs for %d messages', len(X))
    uids = compute_uids(X)

    # Lightweight stratification proxy: number of positive labels (clipped)
    import numpy as np
    pos_counts = Y.sum(axis=1)
    if isinstance(pos_counts, list):
        pos_counts = np.array(pos_counts)
    pos_counts = np.clip(pos_counts, a_min=None, a_max=3)

    logging.info('Splitting into train/eval with test_size=%.2f seed=%d', args.test_size, args.seed)
    indices = list(range(len(uids)))
    _, idx_eval = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=pos_counts
    )

    eval_uids = [uids[i] for i in idx_eval]

    # Create JSON structure with metadata
    json_data = {
        'metadata': {
            'created_date': datetime.now().strftime('%Y-%m-%d'),
            'purpose': 'frozen_evaluation_set',
            'count': len(eval_uids),
            'test_size': args.test_size,
            'random_seed': args.seed,
            'source_db': args.database_filepath,
            'uid_algorithm': 'sha1(message|row_index)'
        },
        'eval_ids': eval_uids
    }

    out_dir = os.path.dirname(args.out_json) or '.'
    os.makedirs(out_dir, exist_ok=True)

    with open(args.out_json, 'w') as f:
        json.dump(json_data, f, indent=2)

    logging.info('Wrote %d eval UIDs to %s', len(eval_uids), args.out_json)
    print(f"Wrote {len(eval_uids)} eval UIDs to {args.out_json}")


if __name__ == '__main__':
    main()


