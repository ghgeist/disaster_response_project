"""
Model health monitoring and metrics utilities.
"""
from __future__ import annotations

import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

from disasterproject.utils.config import BASE_METRICS_PATH, OPT_METRICS_PATH
from disasterproject.utils.metrics_io import read_metrics_csv

from .errors import ModelServiceError
from .model_loader import ModelLoader
from .model_service import ModelService

logger = logging.getLogger(__name__)


def load_metric_frames() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load baseline and optimized metrics DataFrames if available."""
    base_df = read_metrics_csv(BASE_METRICS_PATH)
    opt_df = read_metrics_csv(OPT_METRICS_PATH)
    return base_df, opt_df


def extract_perf_triplet(
    base_df: pd.DataFrame, opt_df: pd.DataFrame
) -> Tuple[Dict[str, List[float]], List[str]]:
    """
    Build metrics dict {'precision':[base,opt], 'recall':[base,opt], 'f1':[base,opt]} and labels.
    Prefers positive class '1'; falls back to a 'macro' row; finally first row.
    Values are interpreted as percentages if >1, otherwise multiplied by 100.
    """
    if base_df is None or opt_df is None:
        raise ValueError("Both base_df and opt_df are required to extract performance triplet")

    def select_row(df: pd.DataFrame) -> pd.Series:
        candidates = df[df.get("output_class", "").astype(str).isin(["1", "positive", "pos"])].head(1)
        if candidates.empty and "output_class" in df.columns:
            try:
                candidates = df[df["output_class"].str.contains("macro", case=False, na=False)].head(1)
            except (AttributeError, KeyError):
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
        except (TypeError, ValueError):
            return 0.0

    metrics: Dict[str, List[float]] = {
        "precision": [to_percent(base_precision), to_percent(opt_precision)],
        "recall": [to_percent(base_recall), to_percent(opt_recall)],
        "f1": [to_percent(base_f1), to_percent(opt_f1)],
    }
    labels: List[str] = ["Baseline Model", "Optimized Model"]
    return metrics, labels


class ModelHealthMonitor:
    """Monitor model health, performance, and system metrics."""

    def __init__(self, base_dir: Path = None, model_service: ModelService = None):
        """Initialize monitor with base directory and optional model service."""
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent.parent
        self.model_dir = self.base_dir / "model"
        self.experiments_dir = self.base_dir / "experiments"
        self.data_dir = self.base_dir / "data" / "04_fct"
        self.model_service = model_service

    def get_model_files(self) -> List[Dict[str, Any]]:
        """Get information about all model files in the system."""
        model_files = []

        if self.model_dir.exists():
            for file_path in self.model_dir.glob("*.pkl"):
                model_files.append(self._get_file_metadata(file_path, "production"))

        if self.experiments_dir.exists():
            for file_path in self.experiments_dir.rglob("*.pkl"):
                if "model" in str(file_path).lower():
                    model_files.append(self._get_file_metadata(file_path, "experimental"))

        return sorted(
            model_files, key=lambda x: x.get("last_modified", datetime.min), reverse=True
        )

    def _get_file_metadata(self, file_path: Path, model_type: str) -> Dict[str, Any]:
        """Get metadata for a single model file."""
        try:
            stat = file_path.stat()
            size_mb = stat.st_size / (1024 * 1024)

            return {
                "name": file_path.name,
                "path": str(file_path),
                "type": model_type,
                "size_mb": round(size_mb, 2),
                "last_modified": datetime.fromtimestamp(stat.st_mtime),
                "last_modified_str": datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "exists": True,
                "loadable": self._test_model_loading(file_path),
            }
        except OSError as error:
            logger.error("Error getting metadata for %s: %s", file_path, error)
            return {
                "name": file_path.name,
                "path": str(file_path),
                "type": model_type,
                "size_mb": 0,
                "last_modified": datetime.min,
                "last_modified_str": "Unknown",
                "exists": False,
                "loadable": False,
                "error": str(error),
            }

    def _test_model_loading(self, file_path: Path) -> bool:
        """Test if a model file can be loaded successfully."""
        try:
            loader = ModelLoader(file_path)
            model = loader.load_local_model()
            return model is not None
        except (
            ModelServiceError,
            OSError,
            joblib.externals.loky.process_executor.TerminatedWorkerError,
            pickle.PickleError,
            EOFError,
            ValueError,
        ):
            return False
        except Exception as error:
            logger.warning("Unexpected error testing model loading for %s: %s", file_path, error)
            return False

    def get_current_model_status(self) -> Dict[str, Any]:
        """Get status of the currently active production model."""
        model_files = list(self.model_dir.glob("*.pkl"))
        if not model_files:
            return {
                "status": "missing",
                "error": "No model files found in model directory",
                "path": str(self.model_dir),
            }

        main_model_path = max(model_files, key=lambda p: p.stat().st_mtime)

        if not main_model_path.exists():
            return {
                "status": "missing",
                "error": "Production model not found",
                "path": str(main_model_path),
            }

        try:
            if self.model_service and self.model_service.model_path == main_model_path:
                start_time = time.time()
                model = self.model_service.load_model()
                load_time = time.time() - start_time
            else:
                temp_service = ModelService(main_model_path)
                start_time = time.time()
                model = temp_service.load_model()
                load_time = time.time() - start_time

            return {
                "status": "healthy",
                "path": str(main_model_path),
                "load_time_ms": round(load_time * 1000, 2),
                "model_type": type(model).__name__,
                "last_loaded": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except ModelServiceError as error:
            logger.error("Error checking current model status: %s", error)
            return {
                "status": "error",
                "error": str(error),
                "path": str(main_model_path),
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics from available data."""
        metrics = {
            "available": False,
            "last_updated": None,
            "baseline_metrics": None,
            "optimized_metrics": None,
            "comparison": None,
        }

        try:
            base_df, opt_df = load_metric_frames()

            if base_df is not None:
                metrics["baseline_metrics"] = self._summarize_metrics(base_df, "baseline")
                metrics["available"] = True
                if BASE_METRICS_PATH.exists():
                    metrics["last_updated"] = datetime.fromtimestamp(
                        BASE_METRICS_PATH.stat().st_mtime
                    ).isoformat()

            if opt_df is not None:
                metrics["optimized_metrics"] = self._summarize_metrics(opt_df, "optimized")

                if metrics["baseline_metrics"]:
                    try:
                        perf_metrics, labels = extract_perf_triplet(base_df, opt_df)
                        metrics["comparison"] = {
                            "precision": perf_metrics["precision"],
                            "recall": perf_metrics["recall"],
                            "f1": perf_metrics["f1"],
                            "labels": labels,
                        }
                    except (KeyError, ValueError) as error:
                        logger.warning("Error extracting performance comparison: %s", error)

        except (OSError, pd.errors.EmptyDataError) as error:
            logger.error("Error loading performance metrics: %s", error)
            metrics["error"] = str(error)

        return metrics

    def _summarize_metrics(self, df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Summarize metrics from a DataFrame."""
        try:
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            summary = {"model_name": model_name, "num_classes": len(df), "metrics": {}}

            metric_cols = ["precision", "recall", "f1-score"]
            for col in metric_cols:
                if col in df.columns:
                    summary["metrics"][col.replace("-", "_")] = {
                        "mean": round(df[col].mean(), 3),
                        "std": round(df[col].std(), 3),
                    }

            return summary

        except (KeyError, TypeError) as error:
            logger.error("Error summarizing metrics for %s: %s", model_name, error)
            return {"model_name": model_name, "error": str(error)}

    def get_prediction_sample(self, model_service=None) -> Dict[str, Any]:
        """Get a sample prediction with confidence scores."""
        if not model_service:
            return {"error": "Model service not available"}

        try:
            sample_messages = [
                "We need urgent medical supplies for earthquake victims",
                "Food and water running low in shelter area",
                "Roads blocked by fallen trees need clearing",
            ]

            results = []
            for msg in sample_messages[:2]:
                try:
                    start_time = time.time()
                    prediction = model_service.predict(msg)
                    prediction_time = time.time() - start_time

                    labels = prediction.get("labels", {})
                    positive_count = sum(1 for v in labels.values() if v == 1)

                    results.append(
                        {
                            "message": msg[:50] + "..." if len(msg) > 50 else msg,
                            "positive_predictions": positive_count,
                            "total_categories": len(labels),
                            "prediction_time_ms": round(prediction_time * 1000, 2),
                        }
                    )
                except ModelServiceError as error:
                    results.append(
                        {
                            "message": msg[:50] + "..." if len(msg) > 50 else msg,
                            "error": str(error),
                        }
                    )

            return {"samples": results, "timestamp": datetime.now().isoformat()}

        except (TypeError, ValueError) as error:
            logger.error("Error getting prediction sample: %s", error)
            return {"error": str(error)}

    def get_comprehensive_health_report(self, model_service=None) -> Dict[str, Any]:
        """Get comprehensive health report for dashboard."""
        return {
            "timestamp": datetime.now().isoformat(),
            "model_files": self.get_model_files(),
            "current_model": self.get_current_model_status(),
            "performance_metrics": self.get_performance_metrics(),
            "prediction_samples": self.get_prediction_sample(model_service),
        }
