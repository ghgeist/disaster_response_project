"""
Metrics loading and extraction utilities.
"""
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from disasterproject.utils.config import (
    BASE_METRICS_PATH,
    OPT_METRICS_PATH,
)
from disasterproject.utils.metrics_io import read_metrics_csv


def load_metric_frames() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load baseline and optimized metrics DataFrames if available."""
    base_df = read_metrics_csv(BASE_METRICS_PATH)
    opt_df = read_metrics_csv(OPT_METRICS_PATH)
    return base_df, opt_df


def extract_perf_triplet(base_df: pd.DataFrame, opt_df: pd.DataFrame) -> Tuple[Dict[str, List[float]], List[str]]:
    """
    Build metrics dict {'precision':[base,opt], 'recall':[base,opt], 'f1':[base,opt]} and labels.
    Prefers positive class '1'; falls back to a 'macro' row; finally first row.
    Values are interpreted as percentages if >1, otherwise multiplied by 100.
    """
    if base_df is None or opt_df is None:
        raise ValueError("Both base_df and opt_df are required to extract performance triplet")

    def select_row(df: pd.DataFrame) -> pd.Series:
        # Try positive class encodings first
        candidates = df[df.get("output_class", "").astype(str).isin(["1", "positive", "pos"])].head(1)
        if candidates.empty and "output_class" in df.columns:
            # Any macro-like row
            try:
                candidates = df[df["output_class"].str.contains("macro", case=False, na=False)].head(1)
            except Exception:
                candidates = pd.DataFrame()
        if candidates.empty and "class" in df.columns:
            candidates = df[df["class"].astype(str).isin(["1", "positive"])].head(1)
        if candidates.empty:
            candidates = df.head(1)
        return candidates.iloc[0]

    base_row = select_row(base_df).to_dict()
    opt_row = select_row(opt_df).to_dict()

    def pick(d: Dict[str, Any], *keys: str, default=None):
        for k in keys:
            if k in d:
                return d[k]
        return default

    base_precision = pick(base_row, "precision", "precision_1", "pos_precision")
    base_recall = pick(base_row, "recall", "recall_1", "pos_recall")
    base_f1 = pick(base_row, "f1-score", "f1_score", "f1", "pos_f1")

    opt_precision = pick(opt_row, "precision", "precision_1", "pos_precision")
    opt_recall = pick(opt_row, "recall", "recall_1", "pos_recall")
    opt_f1 = pick(opt_row, "f1-score", "f1_score", "f1", "pos_f1")

    def to_percent(x) -> float:
        try:
            val = float(x)
            return val * 100.0 if val <= 1.0 else val
        except Exception:
            return 0.0

    metrics: Dict[str, List[float]] = {
        "precision": [to_percent(base_precision), to_percent(opt_precision)],
        "recall": [to_percent(base_recall), to_percent(opt_recall)],
        "f1": [to_percent(base_f1), to_percent(opt_f1)],
    }
    labels: List[str] = ["Baseline Model", "Optimized Model"]
    return metrics, labels
