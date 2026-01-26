"""
Model health monitoring service.
"""
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from .model_service import ModelService
from .metrics_service import load_metric_frames, extract_perf_triplet
from disasterproject.utils.config import BASE_METRICS_PATH

logger = logging.getLogger(__name__)


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
            logger.error("Error getting metadata for %s: %s", file_path, e)
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
            logger.error("Error checking current model status: %s", e)
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
                        logger.warning("Error extracting performance comparison: %s", e)

        except Exception as e:
            logger.error("Error loading performance metrics: %s", e)
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
            logger.error("Error summarizing metrics for %s: %s", model_name, e)
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
            logger.error("Error getting prediction sample: %s", e)
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
