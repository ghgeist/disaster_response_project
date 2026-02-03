"""Tests for model promotion script to prevent algorithm mismatch issues.

These tests ensure:
1. Algorithm type is correctly detected from model files
2. Correct filename is generated based on algorithm type
3. Model file integrity is verified after copying
4. MODEL_INFO.json includes algorithm metadata
5. Promotion flow works correctly for both RF and LR models
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

# Add scripts to path for imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_PATH = PROJECT_ROOT / 'scripts' / '07_operations'
sys.path.insert(0, str(SCRIPTS_PATH))

# pylint: disable=import-error
from promote_model import (  # noqa: E402
    compute_model_hash,
    detect_algorithm_type,
    discover_production_metrics_file,
    promote_model,
    validate_candidate_model,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def rf_model_path(temp_dir):
    """Create a temporary RandomForest model file."""
    model_path = temp_dir / "rf_model.pkl"
    
    # Create a simple RF pipeline
    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=1, random_state=42)))
    ])
    
    # Fit with dummy data
    X = ["test message", "another message"]
    y = [[1, 0], [0, 1]]
    pipeline.fit(X, y)
    
    joblib.dump(pipeline, model_path)
    return model_path


@pytest.fixture
def lr_model_path(temp_dir):
    """Create a temporary LogisticRegression model file."""
    model_path = temp_dir / "lr_model.pkl"
    
    # Create a simple LR pipeline
    pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', MultiOutputClassifier(LogisticRegression(random_state=42, max_iter=100)))
    ])
    
    # Fit with dummy data
    X = ["test message", "another message"]
    y = [[1, 0], [0, 1]]
    pipeline.fit(X, y)
    
    joblib.dump(pipeline, model_path)
    return model_path


@pytest.fixture
def candidate_dir_with_rf_model(temp_dir, rf_model_path):
    """Create a candidate directory with RF model and metadata."""
    candidate_dir = temp_dir / "2025-11-06-rf-candidate"
    candidate_dir.mkdir()
    
    # Copy model
    shutil.copy2(rf_model_path, candidate_dir / "rf_model.pkl")
    
    # Create training log with metrics
    training_log = {
        "performance": {
            "overall_f1": 0.95,
            "micro_f1": 0.70
        }
    }
    (candidate_dir / "training_log.json").write_text(json.dumps(training_log))
    
    return candidate_dir


@pytest.fixture
def candidate_dir_with_lr_model(temp_dir, lr_model_path):
    """Create a candidate directory with LR model and metadata."""
    candidate_dir = temp_dir / "2025-11-06-lr-candidate"
    candidate_dir.mkdir()
    
    # Copy model
    shutil.copy2(lr_model_path, candidate_dir / "lr_model.pkl")
    
    # Create training log with metrics
    training_log = {
        "performance": {
            "overall_f1": 0.94,
            "micro_f1": 0.65
        }
    }
    (candidate_dir / "training_log.json").write_text(json.dumps(training_log))
    
    return candidate_dir


@pytest.fixture
def candidate_dir_with_metrics_csv(temp_dir, lr_model_path):
    """Create a candidate directory with LR model and performance_metrics.csv."""
    candidate_dir = temp_dir / "2025-11-06-lr-with-metrics"
    candidate_dir.mkdir()
    
    # Copy model
    shutil.copy2(lr_model_path, candidate_dir / "lr_model.pkl")
    
    # Create training log with metrics
    training_log = {
        "performance": {
            "overall_f1": 0.94,
            "micro_f1": 0.65
        }
    }
    (candidate_dir / "training_log.json").write_text(json.dumps(training_log))
    
    # Create performance_metrics.csv
    import pandas as pd
    metrics_df = pd.DataFrame({
        'category': ['related', 'related'],
        'output_class': ['0', '1'],
        'precision': [0.8, 0.9],
        'recall': [0.7, 0.85],
        'f1-score': [0.75, 0.875],
        'support': [100.0, 200.0]
    })
    metrics_df.to_csv(candidate_dir / "performance_metrics.csv", index=False)
    
    return candidate_dir


class TestAlgorithmDetection:
    """Test algorithm type detection from model files."""
    
    def test_detect_rf_algorithm(self, rf_model_path):
        """Should correctly detect RandomForest algorithm."""
        algorithm = detect_algorithm_type(rf_model_path)
        assert algorithm == 'rf', f"Expected 'rf', got '{algorithm}'"
    
    def test_detect_lr_algorithm(self, lr_model_path):
        """Should correctly detect LogisticRegression algorithm."""
        algorithm = detect_algorithm_type(lr_model_path)
        assert algorithm == 'lr', f"Expected 'lr', got '{algorithm}'"
    
    def test_detect_algorithm_handles_missing_file(self, temp_dir):
        """Should return 'unknown' for non-existent file."""
        missing_path = temp_dir / "nonexistent.pkl"
        algorithm = detect_algorithm_type(missing_path)
        assert algorithm == 'unknown'
    
    def test_detect_algorithm_handles_invalid_model(self, temp_dir):
        """Should return 'unknown' for invalid/corrupted model file."""
        invalid_path = temp_dir / "invalid.pkl"
        invalid_path.write_text("not a valid pickle file")
        
        # Should handle the error gracefully
        algorithm = detect_algorithm_type(invalid_path)
        assert algorithm == 'unknown'


class TestModelHashVerification:
    """Test model file hash computation and verification."""
    
    def test_compute_model_hash(self, rf_model_path):
        """Should compute consistent hash for same file."""
        hash1 = compute_model_hash(rf_model_path)
        hash2 = compute_model_hash(rf_model_path)
        assert hash1 == hash2, "Hash should be consistent for same file"
        assert len(hash1) == 64, "SHA256 hash should be 64 characters"
    
    def test_hash_different_for_different_models(self, rf_model_path, lr_model_path):
        """Different models should have different hashes."""
        rf_hash = compute_model_hash(rf_model_path)
        lr_hash = compute_model_hash(lr_model_path)
        assert rf_hash != lr_hash, "Different models should have different hashes"


class TestPromotionFlow:
    """Test the full promotion flow."""
    
    def test_promote_rf_model_generates_correct_filename(self, temp_dir, candidate_dir_with_rf_model):
        """RF model promotion should generate filename with 'rf' algorithm code."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        validation_results = validate_candidate_model(candidate_dir_with_rf_model)
        promotion_record = promote_model(candidate_dir_with_rf_model, model_dir, validation_results)
        
        promoted_path = Path(promotion_record['promoted_model'])
        assert 'rf' in promoted_path.name, f"Filename should contain 'rf', got {promoted_path.name}"
        assert promoted_path.name.startswith('disaster_rf_'), f"Filename should start with 'disaster_rf_', got {promoted_path.name}"
        assert promoted_path.exists(), "Promoted model file should exist"
    
    def test_promote_lr_model_generates_correct_filename(self, temp_dir, candidate_dir_with_lr_model):
        """LR model promotion should generate filename with 'lr' algorithm code."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        validation_results = validate_candidate_model(candidate_dir_with_lr_model)
        promotion_record = promote_model(candidate_dir_with_lr_model, model_dir, validation_results)
        
        promoted_path = Path(promotion_record['promoted_model'])
        assert 'lr' in promoted_path.name, f"Filename should contain 'lr', got {promoted_path.name}"
        assert promoted_path.name.startswith('disaster_lr_'), f"Filename should start with 'disaster_lr_', got {promoted_path.name}"
        assert promoted_path.exists(), "Promoted model file should exist"
    
    def test_promoted_model_hash_matches_validation(self, temp_dir, candidate_dir_with_rf_model):
        """Promoted model file should have matching hash from validation."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        validation_results = validate_candidate_model(candidate_dir_with_rf_model)
        expected_hash = validation_results['model_hash']
        
        promotion_record = promote_model(candidate_dir_with_rf_model, model_dir, validation_results)
        promoted_path = Path(promotion_record['promoted_model'])
        
        actual_hash = compute_model_hash(promoted_path)
        assert actual_hash == expected_hash, (
            f"Promoted model hash mismatch!\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual:   {actual_hash}"
        )
    
    def test_model_info_includes_algorithm_metadata(self, temp_dir, candidate_dir_with_rf_model):
        """MODEL_INFO.json should include algorithm and algorithm_name fields."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        validation_results = validate_candidate_model(candidate_dir_with_rf_model)
        promote_model(candidate_dir_with_rf_model, model_dir, validation_results)
        
        model_info_path = model_dir / "MODEL_INFO.json"
        assert model_info_path.exists(), "MODEL_INFO.json should be created"
        
        with open(model_info_path, encoding='utf-8') as f:
            model_info = json.load(f)
        
        assert 'algorithm' in model_info, "MODEL_INFO.json should include 'algorithm' field"
        assert 'algorithm_name' in model_info, "MODEL_INFO.json should include 'algorithm_name' field"
        assert model_info['algorithm'] == 'rf', f"Algorithm should be 'rf', got '{model_info['algorithm']}'"
        assert model_info['algorithm_name'] == 'RandomForest', f"Algorithm name should be 'RandomForest', got '{model_info['algorithm_name']}'"
    
    def test_model_info_includes_lr_algorithm_metadata(self, temp_dir, candidate_dir_with_lr_model):
        """MODEL_INFO.json should correctly identify LR algorithm."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        validation_results = validate_candidate_model(candidate_dir_with_lr_model)
        promote_model(candidate_dir_with_lr_model, model_dir, validation_results)
        
        model_info_path = model_dir / "MODEL_INFO.json"
        with open(model_info_path, encoding='utf-8') as f:
            model_info = json.load(f)
        
        assert model_info['algorithm'] == 'lr', f"Algorithm should be 'lr', got '{model_info['algorithm']}'"
        assert model_info['algorithm_name'] == 'LogisticRegression', f"Algorithm name should be 'LogisticRegression', got '{model_info['algorithm_name']}'"
    
    def test_promotion_record_includes_algorithm_info(self, temp_dir, candidate_dir_with_rf_model):
        """Promotion record should include algorithm information."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        validation_results = validate_candidate_model(candidate_dir_with_rf_model)
        promotion_record = promote_model(candidate_dir_with_rf_model, model_dir, validation_results)
        
        # Check that promotion was successful
        assert promotion_record['status'] == 'promoted'
        assert 'promoted_model' in promotion_record
        
        # Verify the filename in promotion record matches algorithm
        promoted_path = Path(promotion_record['promoted_model'])
        assert 'rf' in promoted_path.name


class TestHashMismatchProtection:
    """Test that hash mismatch errors are caught and reported."""
    
    def test_promotion_fails_on_hash_mismatch(self, temp_dir, candidate_dir_with_rf_model):
        """Promotion should fail if copied file hash doesn't match expected hash."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        validation_results = validate_candidate_model(candidate_dir_with_rf_model)
        
        # Corrupt the hash in validation_results to simulate mismatch
        original_hash = validation_results['model_hash']
        validation_results['model_hash'] = '0' * 64  # Invalid hash
        
        # Promotion should raise an error
        with pytest.raises(ValueError, match="Model file integrity check failed"):
            promote_model(candidate_dir_with_rf_model, model_dir, validation_results)
        
        # Restore original hash for cleanup
        validation_results['model_hash'] = original_hash


class TestIntegrationWithRealModels:
    """Integration tests using actual model files if available."""
    
    def test_detect_algorithm_from_production_rf_model(self):
        """Should detect RF algorithm from actual production model if it exists."""
        prod_model = PROJECT_ROOT / "model" / "disaster_rf_prod_2026-01-22.pkl"
        if prod_model.exists():
            algorithm = detect_algorithm_type(prod_model)
            assert algorithm == 'rf', f"Production RF model should be detected as 'rf', got '{algorithm}'"
        else:
            pytest.skip("Production RF model not found")
    
    def test_detect_algorithm_from_experimental_lr_model(self):
        """Should detect LR algorithm from experimental LR model if it exists."""
        lr_model = PROJECT_ROOT / "experiments" / "experimental_runs" / "2025-11-06-vocab15k-promotion" / "lr_vocab15k_model.pkl"
        if lr_model.exists():
            algorithm = detect_algorithm_type(lr_model)
            assert algorithm == 'lr', f"Experimental LR model should be detected as 'lr', got '{algorithm}'"
        else:
            pytest.skip("Experimental LR model not found")


class TestFilenameGeneration:
    """Test that filenames are generated correctly based on algorithm type."""
    
    def test_rf_filename_format(self, temp_dir, candidate_dir_with_rf_model):
        """RF model should generate filename with correct format."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        validation_results = validate_candidate_model(candidate_dir_with_rf_model)
        promotion_record = promote_model(candidate_dir_with_rf_model, model_dir, validation_results)
        
        filename = Path(promotion_record['promoted_model']).name
        # Format: disaster_{algorithm}_{version}_prod_{date}.pkl
        parts = filename.split('_')
        assert parts[0] == 'disaster'
        assert parts[1] == 'rf'  # Algorithm code
        assert parts[2].startswith('v')  # Version
        assert 'prod' in parts
        assert filename.endswith('.pkl')
    
    def test_lr_filename_format(self, temp_dir, candidate_dir_with_lr_model):
        """LR model should generate filename with correct format."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        validation_results = validate_candidate_model(candidate_dir_with_lr_model)
        promotion_record = promote_model(candidate_dir_with_lr_model, model_dir, validation_results)
        
        filename = Path(promotion_record['promoted_model']).name
        parts = filename.split('_')
        assert parts[0] == 'disaster'
        assert parts[1] == 'lr'  # Algorithm code
        assert parts[2].startswith('v')  # Version
        assert 'prod' in parts
        assert filename.endswith('.pkl')


class TestMetricsFileNaming:
    """Test that performance_metrics.csv uses model-specific naming."""
    
    def test_metrics_file_copied_with_model_specific_naming(self, temp_dir, candidate_dir_with_metrics_csv):
        """Metrics file should be copied with model-specific naming during promotion."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        validation_results = validate_candidate_model(candidate_dir_with_metrics_csv)
        promotion_record = promote_model(candidate_dir_with_metrics_csv, model_dir, validation_results)
        
        # Get the promoted model filename
        promoted_model_path = Path(promotion_record['promoted_model'])
        base_name = promoted_model_path.stem
        
        # Expected metrics file name
        expected_metrics_file = model_dir / f"{base_name}_performance_metrics.csv"
        
        assert expected_metrics_file.exists(), (
            f"Metrics file should exist at {expected_metrics_file}\n"
            f"Promoted model: {promoted_model_path.name}"
        )
        
        # Verify the metrics file content matches the source
        import pandas as pd
        source_metrics = pd.read_csv(candidate_dir_with_metrics_csv / "performance_metrics.csv")
        promoted_metrics = pd.read_csv(expected_metrics_file)
        
        pd.testing.assert_frame_equal(source_metrics, promoted_metrics)
    
    def test_metrics_file_not_copied_if_missing(self, temp_dir, candidate_dir_with_lr_model):
        """Promotion should succeed even if performance_metrics.csv is missing."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        validation_results = validate_candidate_model(candidate_dir_with_lr_model)
        promotion_record = promote_model(candidate_dir_with_lr_model, model_dir, validation_results)
        
        # Promotion should succeed
        assert promotion_record['status'] == 'promoted'
        
        # Metrics file should not exist (wasn't in candidate)
        promoted_model_path = Path(promotion_record['promoted_model'])
        base_name = promoted_model_path.stem
        metrics_file = model_dir / f"{base_name}_performance_metrics.csv"
        
        assert not metrics_file.exists(), "Metrics file should not exist if not in candidate"
    
    def test_discover_production_metrics_file_finds_model_specific_file(self, temp_dir, candidate_dir_with_metrics_csv):
        """discover_production_metrics_file should find model-specific metrics file."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        # Promote model with metrics
        validation_results = validate_candidate_model(candidate_dir_with_metrics_csv)
        promotion_record = promote_model(candidate_dir_with_metrics_csv, model_dir, validation_results)
        
        # Discover metrics file
        discovered_metrics = discover_production_metrics_file(model_dir)
        
        assert discovered_metrics is not None, "Should discover metrics file"
        assert discovered_metrics.exists(), "Discovered metrics file should exist"
        
        # Verify it's the model-specific file
        promoted_model_path = Path(promotion_record['promoted_model'])
        base_name = promoted_model_path.stem
        expected_metrics_file = model_dir / f"{base_name}_performance_metrics.csv"
        
        assert discovered_metrics == expected_metrics_file, (
            f"Should discover model-specific file\n"
            f"  Expected: {expected_metrics_file}\n"
            f"  Found: {discovered_metrics}"
        )
    
    def test_discover_production_metrics_file_falls_back_to_legacy_naming(self, temp_dir):
        """discover_production_metrics_file should fall back to legacy naming if model-specific file missing."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        # Create a production model file
        model_file = model_dir / "disaster_lr_v25-11-06_prod_2025-11-06.pkl"
        model_file.write_bytes(b"fake model data")
        
        # Create legacy metrics file (but not model-specific one)
        legacy_metrics = model_dir / "performance_metrics.csv"
        legacy_metrics.write_text("category,output_class,precision\nrelated,1,0.9")
        
        # Discover should find legacy file
        discovered_metrics = discover_production_metrics_file(model_dir)
        
        assert discovered_metrics is not None, "Should discover legacy metrics file"
        assert discovered_metrics == legacy_metrics, "Should return legacy file when model-specific missing"
    
    def test_discover_production_metrics_file_returns_none_when_no_metrics(self, temp_dir):
        """discover_production_metrics_file should return None when no metrics file exists."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        # Create a production model file but no metrics
        model_file = model_dir / "disaster_lr_v25-11-06_prod_2025-11-06.pkl"
        model_file.write_bytes(b"fake model data")
        
        # Discover should return None
        discovered_metrics = discover_production_metrics_file(model_dir)
        
        assert discovered_metrics is None, "Should return None when no metrics file exists"
    
    def test_discover_production_metrics_file_handles_multiple_models(self, temp_dir):
        """discover_production_metrics_file should use the latest model when multiple exist."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        # Create two model files with different timestamps
        import time
        model1 = model_dir / "disaster_lr_v25-11-05_prod_2025-11-05.pkl"
        model1.write_bytes(b"older model")
        time.sleep(0.1)  # Ensure different mtime
        
        model2 = model_dir / "disaster_lr_v25-11-06_prod_2025-11-06.pkl"
        model2.write_bytes(b"newer model")
        
        # Create metrics file for newer model only
        metrics2 = model_dir / "disaster_lr_v25-11-06_prod_2025-11-06_performance_metrics.csv"
        metrics2.write_text("category,output_class,precision\nrelated,1,0.9")
        
        # Discover should find metrics for newer model
        discovered_metrics = discover_production_metrics_file(model_dir)
        
        assert discovered_metrics is not None
        assert discovered_metrics == metrics2, "Should find metrics for latest model"
