"""
Services for data and model management.
"""
import os
import joblib
import requests
import pandas as pd
import sqlalchemy.exc
from pathlib import Path
from typing import Optional, Any
from sqlalchemy import create_engine
import logging

logger = logging.getLogger(__name__)

# --- Performance metrics helpers (UI deep dive) ---
from typing import Tuple, List, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
FCT_DIR = BASE_DIR / "data" / "04_fct"
BASE_METRICS_PATH = FCT_DIR / "fct_median_metrics_by_output_class_base.csv"
OPT_METRICS_PATH = FCT_DIR / "fct_median_metrics_by_output_class_optimized.csv"


def _read_metrics_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a metrics CSV and normalize column names; return None if missing."""
    try:
        if not path.exists():
            logger.warning(f"Metrics CSV not found: {path}")
            return None
        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if "output_class" in df.columns:
            df["output_class"] = df["output_class"].astype(str)
        return df
    except Exception as exc:
        logger.error(f"Failed reading metrics CSV {path}: {exc}")
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
            logger.info(f"Data loaded successfully from table '{table_name}'")
            return self._df
            
        except (OSError, pd.errors.DatabaseError, sqlalchemy.exc.SQLAlchemyError) as e:
            logger.error(f"Error loading data from database: {e}")
            raise RuntimeError(f"Failed to load data: {e}") from e
    
    def get_data(self) -> pd.DataFrame:
        """Get the loaded data."""
        if self._df is None:
            self.load_data()
        return self._df
    
    def get_category_columns(self) -> list:
        """Get the category column names (assuming they start from column 4)."""
        df = self.get_data()
        return df.columns[4:].tolist()


class ModelService:
    """Service for managing ML model loading and prediction."""
    
    def __init__(self, model_path: Path, gdrive_model_id: Optional[str] = None):
        self.model_path = model_path
        self.gdrive_model_id = gdrive_model_id
        self._model = None
    
    def load_model(self) -> Any:
        """Load the ML model, downloading if necessary."""
        if self._model is not None:
            return self._model
            
        try:
            # Import tokenize function to make it available for unpickling
            # This is required because the pickled models contain references to this function
            import sys
            import os
            import pickle
            
            # Add src to path and import tokenize
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
            from disaster_classifier.data.preprocessor import tokenize
            
            # Make tokenize available in the main module for unpickling
            import __main__
            __main__.tokenize = tokenize
            
            # Ensure model exists locally
            if not self.model_path.exists():
                self._download_model()
            
            # Load the model
            self._model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            return self._model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}") from e
    
    def _download_model(self) -> None:
        """Download model from Google Drive if not available locally."""
        if not self.gdrive_model_id or self.gdrive_model_id.strip() in {'', 'YOUR_FILE_ID', 'YOUR_GOOGLE_DRIVE_FILE_ID'}:
            raise RuntimeError(
                "GDRIVE_MODEL_ID is not set or is using a placeholder. "
                f"Provide a valid Google Drive file ID via the GDRIVE_MODEL_ID env var, "
                f"or place the model at: {self.model_path}"
            )
        
        logger.info("Model not found locally, downloading from Google Drive...")
        
        # Create directory if it doesn't exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temporary file for download
        temp_path = f"{self.model_path}.tmp"
        
        try:
            url = f"https://drive.google.com/uc?export=download&id={self.gdrive_model_id}"
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                
                # Check if response is HTML (Google Drive warning page)
                content_type = r.headers.get('content-type', '')
                if 'text/html' in content_type.lower():
                    raise RuntimeError(
                        "Google Drive returned HTML instead of the model file. "
                        "This usually means the file requires authentication or is too large. "
                        "Please check the GDRIVE_MODEL_ID or download manually."
                    )
                
                # Download to temporary file first
                with open(temp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            
            # Validate the downloaded file
            if os.path.getsize(temp_path) < 1000:  # Model files should be at least 1KB
                raise RuntimeError("Downloaded file is too small, likely corrupted")
            
            # Try to load the model to validate it's not corrupted
            try:
                test_model = joblib.load(temp_path)
                del test_model  # Clean up test load
            except Exception as e:
                raise RuntimeError(f"Downloaded model file is corrupted: {e}") from e
            
            # If validation passes, move temp file to final location
            os.replace(temp_path, self.model_path)
            logger.info("Model downloaded and validated successfully!")
            
        except Exception as e:
            # Clean up temporary file on any error
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass  # Ignore cleanup errors
            
            # Provide helpful error message
            if "timeout" in str(e).lower():
                raise RuntimeError(
                    f"Download timed out. Please check your internet connection and try again. "
                    f"Error: {e}"
                ) from e
            if "corrupted" in str(e).lower():
                raise RuntimeError(
                    f"Download failed due to corruption. Please try again. "
                    f"Error: {e}"
                ) from e
            raise RuntimeError(
                f"Failed to download model: {e}. "
                f"Please check the GDRIVE_MODEL_ID or download manually to: {self.model_path}"
            ) from e
    
    def predict(self, text: str) -> dict:
        """Make a prediction on the given text."""
        if self._model is None:
            self.load_model()
        
        try:
            # Get prediction from model
            classification_labels = self._model.predict([text])[0]
            
            # Get category names (these should ideally come from the data service)
            category_names = [
                'related', 'request', 'offer', 'aid_related', 'medical_help',
                'medical_products', 'search_and_rescue', 'security', 'military',
                'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
                'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
                'transport', 'buildings', 'electricity', 'tools', 'hospitals',
                'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
                'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather',
                'direct_report'
            ]
            
            # Create results dictionary
            results = dict(zip(category_names, classification_labels))
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e
