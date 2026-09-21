#!/usr/bin/env python3
"""
Create a frozen threshold-calibration set (by stable UID) from the SQLite DB.

Calibration UIDs are carved from the non-eval pool only. The frozen eval set
(eval_ids.json) is never modified and remains report-only for threshold work.

Usage:
    python scripts/01_data/create_frozen_cal_ids.py \
        --db data/02_stg/stg_disaster_response.db \
        --eval-ids experiments/experimental_configs/eval_sets/eval_ids.json \
        --out experiments/experimental_configs/eval_sets/cal_ids.json \
        --cal-fraction 0.15 --seed 42
"""

# Standard library imports
import argparse
import json
import logging
import os
import sys
from datetime import datetime

# Third-party imports
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Local imports
from disasterproject.data.loader import load_data
from disasterproject.data.splits import (
    assert_critical_support,
    compute_uids,
    critical_label_positive_support,
    load_uid_list,
    positive_count_strata,
)
from disasterproject.utils.config import CRITICAL_LABELS, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a frozen calibration UID set from the non-eval train pool."
    )
    parser.add_argument(
        "--db",
        dest="database_filepath",
        default="data/02_stg/stg_disaster_response.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--eval-ids",
        dest="eval_ids_path",
        default="experiments/experimental_configs/eval_sets/eval_ids.json",
        help="Path to frozen eval UIDs JSON",
    )
    parser.add_argument(
        "--out",
        dest="out_json",
        default="experiments/experimental_configs/eval_sets/cal_ids.json",
        help="Output JSON path for calibration UIDs",
    )
    parser.add_argument(
        "--cal-fraction",
        dest="cal_fraction",
        type=float,
        default=0.15,
        help="Fraction of non-eval pool held out for calibration (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()
    setup_logging()

    logging.info("Loading data from %s", args.database_filepath)
    X, Y = load_data(args.database_filepath)
    if X is None or Y is None:
        logging.error("Failed to load data. Exiting.")
        sys.exit(1)

    uids = compute_uids(X)
    eval_uids = set(load_uid_list(args.eval_ids_path, "eval_ids"))

    train_pool_indices = [i for i, uid in enumerate(uids) if uid not in eval_uids]
    if not train_pool_indices:
        logging.error("Non-eval pool is empty; cannot carve calibration IDs.")
        sys.exit(1)

    strata = positive_count_strata(Y[train_pool_indices])
    logging.info(
        "Carving cal from non-eval pool: pool=%d, cal_fraction=%.2f, seed=%d",
        len(train_pool_indices),
        args.cal_fraction,
        args.seed,
    )

    _, cal_local_idx = train_test_split(
        train_pool_indices,
        test_size=args.cal_fraction,
        random_state=args.seed,
        stratify=strata,
    )

    cal_uids = [uids[i] for i in cal_local_idx]
    cal_uid_set = set(cal_uids)
    overlap = cal_uid_set & eval_uids
    if overlap:
        logging.error("cal ∩ eval non-empty (%d UIDs); aborting.", len(overlap))
        sys.exit(1)

    Y_cal = Y[cal_local_idx]
    support = critical_label_positive_support(Y_cal)
    print("\ncal positive support:")
    for label in sorted(CRITICAL_LABELS):
        print(f"  {label}: {support[label]}")

    try:
        assert_critical_support(support, context="calibration")
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    train_uids = [
        uid for uid in uids if uid not in eval_uids and uid not in cal_uid_set
    ]
    if set(train_uids) | cal_uid_set | eval_uids != set(uids):
        logging.error("Incomplete UID partition after carving cal_ids.")
        sys.exit(1)

    json_data = {
        "metadata": {
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "purpose": "threshold_calibration_set",
            "count": len(cal_uids),
            "cal_fraction": args.cal_fraction,
            "random_seed": args.seed,
            "source_db": args.database_filepath,
            "source_eval_ids": args.eval_ids_path,
            "uid_algorithm": "sha1(message|row_index)",
            "non_eval_pool_size": len(train_pool_indices),
            "train_residual_size": len(train_uids),
            "critical_label_positive_support": support,
        },
        "cal_ids": cal_uids,
    }

    out_dir = os.path.dirname(args.out_json) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(json_data, handle, indent=2)

    logging.info("Wrote %d cal UIDs to %s", len(cal_uids), args.out_json)
    print(f"\nWrote {len(cal_uids)} cal UIDs to {args.out_json}")
    print(
        f"Partition sizes — train: {len(train_uids)}, "
        f"cal: {len(cal_uids)}, eval: {len(eval_uids)}"
    )


if __name__ == "__main__":
    main()
