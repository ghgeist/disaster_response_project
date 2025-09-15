"""
Services for data and model management.
"""
import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
import requests
import sqlalchemy.exc
from sqlalchemy import create_engine

# Import TARGET_COLUMNS for consistent schema handling
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from disasterproject.utils.config import TARGET_COLUMNS

logger = logging.getLogger(__name__)


class ModelDownloadSkipped(Exception):
    """Exception raised when model download should be skipped."""

BASE_DIR = Path(__file__).resolve().parent.parent
FCT_DIR = BASE_DIR / "data" / "04_fct"
BASE_METRICS_PATH = FCT_DIR / "fct_median_metrics_by_output_class_base.csv"
OPT_METRICS_PATH = FCT_DIR / "fct_median_metrics_by_output_class_optimized.csv"


def _read_metrics_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a metrics CSV and normalize column names; return None if missing."""
    try:
        if not path.exists():
            logger.warning("Metrics CSV not found: %s", path)
            return None
        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if "output_class" in df.columns:
            df["output_class"] = df["output_class"].astype(str)
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        logger.error("File not found or empty metrics CSV %s: %s", path, exc)
        return None
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        logger.error("Parse error in metrics CSV %s: %s", path, exc)
        return None
    except Exception:
        logger.exception("Unexpected error reading metrics CSV %s. See traceback:", path)
        return None


def load_metric_frames() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load baseline and optimized metrics DataFrames if available."""
    base_df = _read_metrics_csv(BASE_METRICS_PATH)
    opt_df = _read_metrics_csv(OPT_METRICS_PATH)
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

    def pick(d: Dict[str, any], *keys: str, default=None):
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


class DataService:
    """Service for managing data loading and operations."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._engine = None
        self._df = None
    
    @property
    def engine(self):
        """Get database engine, creating if necessary."""
        if self._engine is None:
            self._engine = create_engine(self.database_url)
        return self._engine
    
    def load_data(self, table_name: str = 'stg_disaster_response') -> pd.DataFrame:
        """Load data from the database."""
        if self._df is not None:
            return self._df
            
        try:
            self._df = pd.read_sql_table(table_name, self.engine)
            logger.info("Data loaded successfully from table '%s'", table_name)
            return self._df
            
        except (OSError, pd.errors.DatabaseError, sqlalchemy.exc.SQLAlchemyError) as e:
            logger.error("Error loading data from database: %s", e)
            raise RuntimeError(f"Failed to load data: {e}") from e
    
    def get_data(self) -> pd.DataFrame:
        """Get the loaded data."""
        if self._df is None:
            self.load_data()
        return self._df
    
    def get_category_columns(self) -> list:
        """Get the category column names using TARGET_COLUMNS for consistency."""
        # Use TARGET_COLUMNS from config instead of hardcoded assumption about column positions
        df = self.get_data()
        # Return only columns that exist in both TARGET_COLUMNS and the dataframe
        available_columns = set(df.columns)
        return [col for col in TARGET_COLUMNS if col in available_columns]


class ModelService:
    """Service for managing ML model loading and prediction."""
    
    def __init__(self, model_path: Path, gdrive_model_id: Optional[str] = None):
        self.model_path = model_path
        self.gdrive_model_id = gdrive_model_id
        self._model = None
        self._thresholds = None
        self._label_order = None
    
    def load_model(self) -> Any:
        """Load the ML model, downloading if necessary."""
        if self._model is not None:
            return self._model

        try:
            # Environment-aware model loading
            # If in Replit, always try to download fresh models. Otherwise only download if missing.
            should_download = (
                os.getenv('REPLIT_DB_URL') is not None or  # Replit: always refresh
                not self.model_path.exists()               # Local: only if missing
            )

            if should_download:
                try:
                    self._download_model()
                except ModelDownloadSkipped:
                    # Download was skipped because local model exists - this is fine
                    logger.debug("Model download skipped, using existing local model")
                except RuntimeError as e:
                    # Download failed for other reasons
                    if self.model_path.exists():
                        logger.warning("Download failed but using existing local model: %s", e)
                    else:
                        # No local model and download failed - re-raise
                        raise
            
            # Try standard loading first
            self._model = joblib.load(self.model_path)
            logger.info("Model loaded successfully from %s", self.model_path)
            
            # Attempt to load thresholds and label order co-located with model
            self._load_artifacts()
            return self._model
            
        except (FileNotFoundError, OSError) as e:
            logger.error("Model file not found or inaccessible: %s", e)
            raise RuntimeError(f"Model file not found: {e}") from e
        except (joblib.externals.loky.process_executor.TerminatedWorkerError, pickle.PickleError) as e:
            logger.error("Model file corrupted or incompatible: %s", e)
            raise RuntimeError(f"Model file is corrupted: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error loading model. See traceback:")
            raise RuntimeError(f"Failed to load model: {e}") from e
    
    def _download_model(self) -> None:
        """Download model from Google Drive if not available locally."""
        self._validate_gdrive_config()
        logger.info("Model not found locally, downloading from Google Drive...")
        
        # Prepare for download
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = f"{self.model_path}.tmp"
        
        try:
            self._perform_download(temp_path)
            self._validate_downloaded_file(temp_path)
            self._finalize_download(temp_path)
            logger.info("Model downloaded and validated successfully!")

        except requests.exceptions.RequestException as e:
            self._cleanup_temp_file(temp_path)
            logger.error("Network error downloading model: %s", e)
            raise RuntimeError(f"Network error downloading model: {e}") from e
        except Exception as e:
            self._cleanup_temp_file(temp_path)
            self._handle_download_error(e)
    
    def _validate_gdrive_config(self) -> None:
        """Validate Google Drive configuration before attempting download."""
        if not self.gdrive_model_id or self.gdrive_model_id.strip() in {'', 'YOUR_FILE_ID', 'YOUR_GOOGLE_DRIVE_FILE_ID'}:
            if self.model_path.exists():
                logger.info("Model found at %s, skipping download", self.model_path)
                raise ModelDownloadSkipped("Local model exists, skipping download")
            raise RuntimeError(
                "GDRIVE_MODEL_ID is not set or is using a placeholder. "
                f"Provide a valid Google Drive file ID via the GDRIVE_MODEL_ID env var, "
                f"or place the model at: {self.model_path}"
            )
    
    def _perform_download(self, temp_path: str) -> None:
        """Perform the actual download from Google Drive."""
        url = f"https://drive.google.com/uc?export=download&id={self.gdrive_model_id}"
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            self._validate_response_content_type(r)
            self._write_download_to_file(r, temp_path)
    
    def _validate_response_content_type(self, response) -> None:
        """Validate that the response contains the expected file content."""
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type.lower():
            raise RuntimeError(
                "Google Drive returned HTML instead of the model file. "
                "This usually means the file requires authentication or is too large. "
                "Please check the GDRIVE_MODEL_ID or download manually."
            )
    
    def _write_download_to_file(self, response, temp_path: str) -> None:
        """Write the downloaded content to a temporary file."""
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    
    def _validate_downloaded_file(self, temp_path: str) -> None:
        """Validate the downloaded file size and integrity."""
        if os.path.getsize(temp_path) < 1000:  # Model files should be at least 1KB
            raise RuntimeError("Downloaded file is too small, likely corrupted")
        
        try:
            test_model = joblib.load(temp_path)
            del test_model  # Clean up test load
        except Exception as e:
            raise RuntimeError(f"Downloaded model file is corrupted: {e}") from e
    
    def _finalize_download(self, temp_path: str) -> None:
        """Move the temporary file to the final location."""
        os.replace(temp_path, self.model_path)
    
    def _cleanup_temp_file(self, temp_path: str) -> None:
        """Clean up temporary file on error."""
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass  # Ignore cleanup errors
    
    def _handle_download_error(self, error: Exception) -> None:
        """Handle download errors with appropriate error messages."""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            raise RuntimeError(
                f"Download timed out. Please check your internet connection and try again. "
                f"Error: {error}"
            ) from error
        if "corrupted" in error_str:
            raise RuntimeError(
                f"Download failed due to corruption. Please try again. "
                f"Error: {error}"
            ) from error
        raise RuntimeError(
            f"Failed to download model: {error}. "
            f"Please check the GDRIVE_MODEL_ID or download manually to: {self.model_path}"
        ) from error
    
    def predict(self, text: str) -> dict:
        """Make a prediction on the given text using per-label thresholds when available."""
        if self._model is None:
            self.load_model()
        
        try:
            category_names = self._get_label_order()
            probs: List[float] = []
            prob_dict: Dict[str, float] = {}
            # Try probability-based thresholding
            try:
                proba = self._model.predict_proba([text])
                # predict_proba for MultiOutput returns a list of arrays, one per label
                # Each array shape: (n_samples, n_classes); we want probability of positive class
                if isinstance(proba, list):
                    # Handle case where model was trained on fewer categories than expected
                    # (e.g., some categories had no positive examples and were dropped)
                    model_output_count = len(proba)
                    expected_count = len(category_names)

                    if model_output_count != expected_count:
                        logger.warning(
                            "Model output count (%d) != expected count (%d). Model may have been trained on a subset of categories.",
                            model_output_count,
                            expected_count,
                        )
                        # Create a mapping that ensures all expected categories are handled
                        active_categories, category_mapping = self._create_category_mapping(
                            category_names, model_output_count
                        )
                    else:
                        active_categories = category_names
                        category_mapping = {i: i for i in range(len(category_names))}

                    for idx, p in enumerate(proba):
                        if p.shape[1] == 1:
                            # Single column: assume it's the positive class probability
                            prob_val = p[:, 0][0]
                            probs.append(prob_val)
                            category_name = active_categories[idx] if idx < len(active_categories) else f"unknown_{idx}"
                            logger.debug(
                                "Label %d (%s): single column prob=%.4f",
                                idx,
                                category_name,
                                prob_val,
                            )
                        elif p.shape[1] == 2:
                            # Two columns: assume class 1 is positive (standard binary classification)
                            prob_val = p[:, 1][0]
                            probs.append(prob_val)
                            category_name = active_categories[idx] if idx < len(active_categories) else f"unknown_{idx}"
                            logger.debug(
                                "Label %d (%s): two columns prob=%.4f (class 1)",
                                idx,
                                category_name,
                                prob_val,
                            )
                        else:
                            # Unexpected number of columns
                            logger.warning(
                                "Unexpected predict_proba shape %s for label %d, falling back to predict",
                                p.shape,
                                idx,
                            )
                            raise TypeError(f"Unexpected predict_proba shape {p.shape}")

                    # Create final results with all expected categories
                    category_names, classification_labels, prob_dict = self._map_model_outputs_to_categories(
                        category_names, active_categories, probs, category_mapping
                    )
                else:
                    # Some wrappers may return ndarray; fallback to simple predict
                    raise TypeError("Unexpected predict_proba output; using predict fallback")
            except Exception as prob_exc:
                logger.warning(
                    "Probability path failed (%s); falling back to default predict",
                    prob_exc,
                )
                raw_predictions = self._model.predict([text])[0]
                probs = []

                # Apply category padding logic for predict fallback
                model_output_count = len(raw_predictions)
                expected_count = len(category_names)

                if model_output_count != expected_count:
                    logger.warning(
                        "Predict fallback: Model output count (%d) != expected count (%d)",
                        model_output_count,
                        expected_count,
                    )

                    # Pad or truncate predictions to match expected categories
                    if model_output_count < expected_count:
                        # Model outputs fewer categories - pad with zeros
                        classification_labels = list(raw_predictions) + [0] * (expected_count - model_output_count)
                        logger.info(
                            "Padded %d missing categories with 0",
                            expected_count - model_output_count,
                        )
                    else:
                        # Model outputs more categories - truncate to expected count
                        classification_labels = list(raw_predictions[:expected_count])
                        logger.info(
                            "Truncated %d extra categories",
                            model_output_count - expected_count,
                        )
                else:
                    classification_labels = raw_predictions

                prob_dict = {}

            # Create final results dictionary
            results = dict(zip(category_names, classification_labels))
            if not prob_dict:
                prob_dict = dict(zip(category_names, probs)) if probs else {}
            
            return {"labels": results, "probabilities": prob_dict}
            
        except (ValueError, AttributeError) as e:
            logger.error("Model prediction input error: %s", e)
            raise RuntimeError(f"Invalid input for prediction: {e}") from e
        except (OSError, FileNotFoundError) as e:
            logger.error("Model file access error during prediction: %s", e)
            raise RuntimeError(f"Model file access failed: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error making prediction. See traceback:")
            raise RuntimeError(f"Prediction failed: {e}") from e

    def _load_artifacts(self) -> None:
        """Load thresholds and label_order artifacts from the model directory if present."""
        try:
            model_dir = self.model_path.parent
            model_stem = self.model_path.stem  # Get filename without .pkl extension
            
            # Try standardized naming first, then fall back to legacy naming
            thresholds_candidates = [
                model_dir / f"{model_stem}_thresholds.json",  # Standardized
                model_dir / "thresholds.json"                 # Legacy
            ]
            
            label_order_candidates = [
                model_dir / f"{model_stem}_labels.json",      # Standardized  
                model_dir / "label_order.json"                # Legacy
            ]
            
            # Load thresholds
            self._thresholds = None
            for thresholds_path in thresholds_candidates:
                if thresholds_path.exists():
                    with open(thresholds_path, "r", encoding="utf-8") as f:
                        self._thresholds = json.load(f)
                    logger.info("Loaded thresholds from %s", thresholds_path.name)
                    break
                    
            # Load label order
            self._label_order = None
            for label_order_path in label_order_candidates:
                if label_order_path.exists():
                    with open(label_order_path, "r", encoding="utf-8") as f:
                        self._label_order = json.load(f)
                    logger.info("Loaded label order from %s", label_order_path.name)
                    break
        except Exception as exc:
            logger.warning("Failed loading model artifacts (thresholds/label_order): %s", exc)
            self._thresholds = None
            self._label_order = None

    def _get_label_order(self) -> list:
        """Return label order from artifact if present, else fallback to hardcoded order."""
        if isinstance(self._label_order, list) and self._label_order:
            return self._label_order
        return [
            'related', 'request', 'offer', 'aid_related', 'medical_help',
            'medical_products', 'search_and_rescue', 'security', 'military',
            'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
            'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
            'transport', 'buildings', 'electricity', 'tools', 'hospitals',
            'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
            'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather',
            'direct_report'
        ]

    def _get_thresholds_map(self) -> dict:
        """Return thresholds map; if missing, return defaults for 8 high-impact labels at 0.5."""
        # Default thresholds map
        default = {}
        target_labels = {
            'medical_help', 'search_and_rescue', 'water', 'food', 'shelter',
            'hospitals', 'security', 'weather_related'
        }
        for name in self._get_label_order():
            default[name] = 0.5 if name in target_labels else 0.5
        if isinstance(self._thresholds, dict) and self._thresholds:
            # Merge, favoring stored thresholds
            merged = {**default, **self._thresholds}
            return merged
        return default

    def _create_category_mapping(self, expected_categories: List[str], model_output_count: int) -> Tuple[List[str], Dict[int, int]]:
        """
        Create a mapping between model outputs and expected categories.
        
        This handles the case where the model was trained on a subset of categories.
        It attempts to map model outputs to the most likely corresponding expected categories.
        
        Args:
            expected_categories: List of all expected category names
            model_output_count: Number of outputs the model actually produces
            
        Returns:
            Tuple of (active_categories, category_mapping) where:
            - active_categories: Categories that the model actually outputs
            - category_mapping: Dict mapping model output index to expected category index
        """
        if model_output_count >= len(expected_categories):
            # Model has more or equal outputs than expected - use first N expected categories
            active_categories = expected_categories[:model_output_count]
            category_mapping = {i: i for i in range(model_output_count)}
            logger.info(f"Model has {model_output_count} outputs, using first {len(active_categories)} expected categories")
        else:
            # Model has fewer outputs - try to map to most relevant expected categories
            # For now, use the first N expected categories as a conservative approach
            # In a more sophisticated implementation, this could use feature importance
            # or training metadata to determine which categories were actually used
            active_categories = expected_categories[:model_output_count]
            category_mapping = {i: i for i in range(model_output_count)}
            
            logger.warning(
                f"Model has {model_output_count} outputs but {len(expected_categories)} expected categories. "
                f"Using first {model_output_count} expected categories. "
                f"Consider updating the model or expected categories to match."
            )
        
        return active_categories, category_mapping

    def _map_model_outputs_to_categories(
        self, 
        expected_categories: List[str], 
        active_categories: List[str], 
        model_probs: List[float], 
        category_mapping: Dict[int, int]
    ) -> Tuple[List[str], List[int], Dict[str, float]]:
        """
        Map model outputs back to all expected categories.
        
        Args:
            expected_categories: All expected category names
            active_categories: Categories that the model actually outputs
            model_probs: Probabilities from model outputs
            category_mapping: Mapping from model output index to expected category index
            
        Returns:
            Tuple of (final_categories, final_labels, final_probs) where:
            - final_categories: All expected categories
            - final_labels: Classification labels for all expected categories
            - final_probs: Probabilities for all expected categories
        """
        thresholds = self._get_thresholds_map()
        final_labels = []
        final_probs = {}
        
        # Initialize all expected categories
        for i, category_name in enumerate(expected_categories):
            # Check if this category was in the model output
            model_idx = None
            for model_output_idx, expected_idx in category_mapping.items():
                if expected_idx == i and model_output_idx < len(model_probs):
                    model_idx = model_output_idx
                    break
            
            if model_idx is not None:
                # Category was in model output - use actual probability
                prob_val = model_probs[model_idx]
                threshold = thresholds.get(category_name, 0.5)
                label = 1 if prob_val >= threshold else 0
                final_probs[category_name] = prob_val
            else:
                # Category was not in model output - set to 0 (no prediction)
                prob_val = 0.0
                label = 0
                final_probs[category_name] = prob_val
                logger.debug(f"Category '{category_name}' not in model output, setting to 0")
            
            final_labels.append(label)
        
        return expected_categories, final_labels, final_probs


class ModelHealthMonitor:
    """Monitor model health, performance, and system metrics."""
    
    def __init__(self, base_dir: Path = None, model_service: ModelService = None):
        """Initialize monitor with base directory and optional model service."""
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.model_dir = self.base_dir / "model"
        self.experiments_dir = self.base_dir / "experiments"
        self.data_dir = self.base_dir / "data" / "04_fct"
        self.model_service = model_service
        
    def get_model_files(self) -> List[Dict[str, Any]]:
        """Get information about all model files in the system."""
        model_files = []
        
        # Check main model directory
        if self.model_dir.exists():
            for file_path in self.model_dir.glob("*.pkl"):
                model_files.append(self._get_file_metadata(file_path, "production"))
                
        # Check experiments directory for models
        if self.experiments_dir.exists():
            for file_path in self.experiments_dir.rglob("*.pkl"):
                if "model" in str(file_path).lower():
                    model_files.append(self._get_file_metadata(file_path, "experimental"))
                
        return sorted(model_files, key=lambda x: x.get('last_modified', datetime.min), reverse=True)
    
    def _get_file_metadata(self, file_path: Path, model_type: str) -> Dict[str, Any]:
        """Get metadata for a single model file."""
        try:
            stat = file_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            
            return {
                'name': file_path.name,
                'path': str(file_path),
                'type': model_type,
                'size_mb': round(size_mb, 2),
                'last_modified': datetime.fromtimestamp(stat.st_mtime),
                'last_modified_str': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'exists': True,
                'loadable': self._test_model_loading(file_path)
            }
        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {e}")
            return {
                'name': file_path.name,
                'path': str(file_path),
                'type': model_type,
                'size_mb': 0,
                'last_modified': datetime.min,
                'last_modified_str': 'Unknown',
                'exists': False,
                'loadable': False,
                'error': str(e)
            }
    
    def _test_model_loading(self, file_path: Path) -> bool:
        """Test if a model file can be loaded successfully."""
        try:
            model = joblib.load(file_path)
            return model is not None
        except Exception:
            return False
    
    def get_current_model_status(self) -> Dict[str, Any]:
        """Get status of the currently active production model."""
        # Find the most recent model file in the model directory
        model_files = list(self.model_dir.glob("*.pkl"))
        if not model_files:
            return {
                'status': 'missing',
                'error': 'No model files found in model directory',
                'path': str(self.model_dir)
            }
        
        # Get the most recent model file
        main_model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        
        if not main_model_path.exists():
            return {
                'status': 'missing',
                'error': 'Production model not found',
                'path': str(main_model_path)
            }
        
        try:
            # Use ModelService if available, otherwise load directly
            if self.model_service and self.model_service.model_path == main_model_path:
                start_time = time.time()
                model = self.model_service.load_model()
                load_time = time.time() - start_time
            else:
                # Create a temporary ModelService for loading
                temp_service = ModelService(main_model_path)
                start_time = time.time()
                model = temp_service.load_model()
                load_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'path': str(main_model_path),
                'load_time_ms': round(load_time * 1000, 2),
                'model_type': type(model).__name__,
                'last_loaded': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error checking current model status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'path': str(main_model_path)
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics from available data."""
        metrics = {
            'available': False,
            'last_updated': None,
            'baseline_metrics': None,
            'optimized_metrics': None,
            'comparison': None
        }
        
        try:
            # Use existing metric loading functions
            base_df, opt_df = load_metric_frames()
            
            if base_df is not None:
                metrics['baseline_metrics'] = self._summarize_metrics(base_df, 'baseline')
                metrics['available'] = True
                base_path = BASE_METRICS_PATH
                if base_path.exists():
                    metrics['last_updated'] = datetime.fromtimestamp(base_path.stat().st_mtime).isoformat()
            
            if opt_df is not None:
                metrics['optimized_metrics'] = self._summarize_metrics(opt_df, 'optimized')
                
                # Compare if both available
                if metrics['baseline_metrics']:
                    try:
                        perf_metrics, labels = extract_perf_triplet(base_df, opt_df)
                        metrics['comparison'] = {
                            'precision': perf_metrics['precision'],
                            'recall': perf_metrics['recall'],
                            'f1': perf_metrics['f1'],
                            'labels': labels
                        }
                    except Exception as e:
                        logger.warning(f"Error extracting performance comparison: {e}")
            
        except Exception as e:
            logger.error(f"Error loading performance metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _summarize_metrics(self, df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Summarize metrics from a DataFrame."""
        try:
            # Clean column names
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            
            summary = {
                'model_name': model_name,
                'num_classes': len(df),
                'metrics': {}
            }
            
            # Calculate averages for key metrics
            metric_cols = ['precision', 'recall', 'f1-score']
            for col in metric_cols:
                if col in df.columns:
                    summary['metrics'][col.replace('-', '_')] = {
                        'mean': round(df[col].mean(), 3),
                        'std': round(df[col].std(), 3)
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing metrics for {model_name}: {e}")
            return {'model_name': model_name, 'error': str(e)}
    
    def get_prediction_sample(self, model_service=None) -> Dict[str, Any]:
        """Get a sample prediction with confidence scores."""
        if not model_service:
            return {'error': 'Model service not available'}
        
        try:
            # Sample disaster messages for testing
            sample_messages = [
                "We need urgent medical supplies for earthquake victims",
                "Food and water running low in shelter area",
                "Roads blocked by fallen trees need clearing"
            ]
            
            results = []
            for msg in sample_messages[:2]:  # Limit to 2 for performance
                try:
                    start_time = time.time()
                    prediction = model_service.predict(msg)
                    prediction_time = time.time() - start_time
                    
                    # Count positive predictions from the labels dictionary
                    labels = prediction.get('labels', {})
                    positive_count = sum(1 for v in labels.values() if v == 1)
                    
                    results.append({
                        'message': msg[:50] + "..." if len(msg) > 50 else msg,
                        'positive_predictions': positive_count,
                        'total_categories': len(labels),
                        'prediction_time_ms': round(prediction_time * 1000, 2)
                    })
                except Exception as e:
                    results.append({
                        'message': msg[:50] + "..." if len(msg) > 50 else msg,
                        'error': str(e)
                    })
            
            return {
                'samples': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction sample: {e}")
            return {'error': str(e)}
    
    def get_comprehensive_health_report(self, model_service=None) -> Dict[str, Any]:
        """Get comprehensive health report for dashboard."""
        return {
            'timestamp': datetime.now().isoformat(),
            'model_files': self.get_model_files(),
            'current_model': self.get_current_model_status(),
            'performance_metrics': self.get_performance_metrics(),
            'prediction_samples': self.get_prediction_sample(model_service)
        }
