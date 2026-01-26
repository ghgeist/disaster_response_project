"""
Model service for managing ML model loading and prediction.
"""
import hashlib
import json
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
from sklearn.exceptions import InconsistentVersionWarning

from .exceptions import ModelServiceError

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing ML model loading and prediction."""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model = None
        self._thresholds = None
        self._label_order = None
    
    def load_model(self) -> Any:
        """Load the ML model from local file."""
        if self._model is not None:
            return self._model

        try:
            if not self.model_path.exists():
                raise ModelServiceError(
                    f"Model file not found at {self.model_path}. "
                    "Please ensure the model file exists in the model/ directory."
                )
            
            # Load model from local file
            self._model = self._load_model_with_version_check(self.model_path)
            logger.info("Model loaded successfully from %s", self.model_path)
            
            # Attempt to load thresholds and label order co-located with model
            self._load_artifacts()
            self._log_model_diagnostics()
            return self._model
            
        except (FileNotFoundError, OSError) as error:
            logger.error("Model file not found or inaccessible: %s", error)
            raise ModelServiceError("Model file not found.") from error
        except (joblib.externals.loky.process_executor.TerminatedWorkerError, pickle.PickleError) as error:
            logger.error("Model file corrupted or incompatible: %s", error)
            raise ModelServiceError("Model file is corrupted.") from error
        except Exception as error:
            logger.exception("Unexpected error loading model from %s", self.model_path)
            raise ModelServiceError("Failed to load model.") from error
    
    def _load_model_with_version_check(self, file_path: Path) -> Any:
        """Load a model and raise if scikit-learn version mismatch is detected."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InconsistentVersionWarning)
            model = joblib.load(file_path)

        for warning in caught:
            if issubclass(warning.category, InconsistentVersionWarning):
                logger.error("scikit-learn version mismatch detected: %s", warning.message)
                raise ModelServiceError(
                    "Model was trained with a different scikit-learn version. "
                    "Retrain the model or install a matching scikit-learn version."
                )

        return model

    def _log_model_diagnostics(self) -> None:
        """Log model metadata and fitted-vectorizer checks for debugging."""
        try:
            model_size_bytes = self.model_path.stat().st_size
        except OSError:
            model_size_bytes = None

        model_sha256 = None
        try:
            with open(self.model_path, "rb") as f:
                model_sha256 = hashlib.sha256(f.read()).hexdigest()
        except OSError as error:
            logger.warning("Failed to read model file for hash: %s", error)

        tfidf_fitted = None
        try:
            tfidf = None
            if hasattr(self._model, "named_steps"):
                tfidf = self._model.named_steps.get("tfidf")
            if tfidf is not None:
                tfidf_fitted = hasattr(tfidf, "idf_")
        except Exception as error:
            logger.warning("Failed to inspect tfidf fitted state: %s", error)

        logger.info(
            "Model diagnostics: path=%s size_bytes=%s sha256=%s tfidf_fitted=%s",
            self.model_path,
            model_size_bytes,
            model_sha256,
            tfidf_fitted,
        )
    
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

                    # Get the MultiOutputClassifier from the pipeline to access classes
                    multi_output_clf = None
                    if hasattr(self._model, 'named_steps') and 'clf' in self._model.named_steps:
                        multi_output_clf = self._model.named_steps['clf']
                    
                    for idx, p in enumerate(proba):
                        if p.shape[1] == 1:
                            # Single column: degenerate classifier (only one class learned)
                            # Check which class is present to determine correct probability
                            if multi_output_clf is not None and hasattr(multi_output_clf, 'classes_') and idx < len(multi_output_clf.classes_):
                                classes = multi_output_clf.classes_[idx]
                                if len(classes) == 1 and classes[0] == 0:
                                    # Only class 0 present, probability of class 1 is 0
                                    prob_val = 0.0
                                elif len(classes) == 1 and classes[0] == 1:
                                    # Only class 1 present, probability of class 1 is 1
                                    prob_val = 1.0
                                else:
                                    # Fallback (shouldn't happen)
                                    prob_val = 0.0
                            else:
                                # Fallback if class info not available (assume class 0)
                                prob_val = 0.0
                            probs.append(prob_val)
                            category_name = active_categories[idx] if idx < len(active_categories) else f"unknown_{idx}"
                            logger.debug(
                                "Label %d (%s): degenerate classifier detected, positive prob set to %.1f",
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
            
        except (ValueError, AttributeError) as error:
            logger.error("Model prediction input error: %s", error)
            raise ModelServiceError("Invalid input for prediction.") from error
        except (OSError, FileNotFoundError) as error:
            logger.error("Model file access error during prediction: %s", error)
            raise ModelServiceError("Model file access failed.") from error
        except Exception as error:
            logger.exception("Unexpected error during prediction for model %s", self.model_path)
            raise ModelServiceError("Prediction failed.") from error

    def _load_artifacts(self) -> None:
        """Load thresholds and label_order artifacts from the model directory if present."""
        try:
            model_dir = self.model_path.parent
            model_stem = self.model_path.stem  # Get filename without .pkl extension
            
            # Standard naming: {model_stem}_thresholds.json
            # Fallback to legacy names for backward compatibility
            thresholds_candidates = [
                model_dir / f"{model_stem}_thresholds.json",       # Standardized (preferred)
                model_dir / "optimized_critical_thresholds.json",  # Legacy: optimized critical thresholds
                model_dir / "optimized_all_thresholds.json",        # Legacy: optimized all thresholds
                model_dir / "thresholds.json"                       # Legacy: F2-optimized thresholds
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
                        loaded_data = json.load(f)
                    
                    # Handle nested structure from optimized_critical_thresholds.json
                    # Format: {"thresholds": {...}, "critical_only": {...}, "metadata": {...}}
                    if isinstance(loaded_data, dict) and "thresholds" in loaded_data:
                        # Extract the thresholds dict from nested structure
                        self._thresholds = loaded_data["thresholds"]
                        logger.info("Loaded optimized thresholds from %s (target recall: %s)", 
                                  thresholds_path.name,
                                  loaded_data.get("metadata", {}).get("target_recall", "unknown"))
                    else:
                        # Flat structure (legacy format)
                        self._thresholds = loaded_data
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

    def get_thresholds_map(self) -> dict:
        """
        Return thresholds map; if missing, return smart defaults based on category type.
        
        Default thresholds are optimized based on category importance:
        - Critical categories: Lower thresholds (0.01-0.43) for better recall
        - Non-critical categories: Standard 0.5 threshold
        
        These defaults are based on optimization results from 2025-11-04 session.
        """
        from disasterproject.utils.config import CRITICAL_LABELS
        
        # Smart defaults based on optimization results (2025-11-04)
        # Critical categories need much lower thresholds to achieve good recall
        critical_defaults = {
            'hospitals': 0.014,           # 1.4% - extremely low for life-safety
            'security': 0.020,            # 2.0% - very low for emergencies
            'search_and_rescue': 0.033,   # 3.3% - low for urgent needs
            'medical_products': 0.095,   # 9.5% - low for medical supplies
            'medical_help': 0.124,        # 12.4% - low for medical emergencies
            'shelter': 0.240,             # 24.0% - moderate for shelter needs
            'water': 0.362,               # 36.2% - moderate for water needs
            'food': 0.431,                # 43.1% - moderate-high for food needs
        }
        
        # Build default map
        default = {}
        for name in self._get_label_order():
            if name in CRITICAL_LABELS and name in critical_defaults:
                # Use optimized default for critical categories
                default[name] = critical_defaults[name]
            else:
                # Standard 0.5 for non-critical categories
                default[name] = 0.5
        
        # Merge with loaded thresholds (loaded thresholds take priority)
        if isinstance(self._thresholds, dict) and self._thresholds:
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
            logger.info(
                "Model has %d outputs, using first %d expected categories",
                model_output_count,
                len(active_categories),
            )
        else:
            # Model has fewer outputs - try to map to most relevant expected categories
            # For now, use the first N expected categories as a conservative approach
            # In a more sophisticated implementation, this could use feature importance
            # or training metadata to determine which categories were actually used
            active_categories = expected_categories[:model_output_count]
            category_mapping = {i: i for i in range(model_output_count)}

            logger.warning(
                "Model has %d outputs but %d expected categories. Using first %d expected categories."
                " Consider updating the model or expected categories to match.",
                model_output_count,
                len(expected_categories),
                model_output_count,
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
        thresholds = self.get_thresholds_map()
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
                logger.debug("Category %s not in model output, setting to 0", category_name)
            
            final_labels.append(label)
        
        return expected_categories, final_labels, final_probs
