---
title: "Critical Review: ML Performance Improvement Plan"
date: "2025-11-04"
status: "active"
tags: ["review", "ml", "execution-plan", "risk-analysis"]
author: "Review Agent"
related: ["2025-11-04-model-performance-improvement-plan.md"]
priority: "CRITICAL"
---

# Critical Review: ML Performance Improvement Plan

## 🚨 CRITICAL ISSUES - Fix Before Execution

### Issue 1: Incorrect Class Weight Implementation
**Problem**: The plan suggests setting class weights AFTER fitting (Method 2), which won't affect the already-trained model.

**Current (WRONG)**:
```python
clf.fit(X, y)
for i, est in enumerate(clf.estimators_):
    est.class_weight = weights_per_label[i]  # This won't work!
```

**Correct Implementation**:
```python
# Create individual estimators with proper weights BEFORE fitting
from sklearn.multioutput import MultiOutputClassifier

def create_weighted_multioutput_classifier(base_estimator_class, class_weights_list, **kwargs):
    estimators = []
    for weights in class_weights_list:
        est = base_estimator_class(class_weight=weights, **kwargs)
        estimators.append(est)
    
    # Custom MultiOutputClassifier that uses pre-configured estimators
    return CustomMultiOutputClassifier(estimators)
```

### Issue 2: Missing Critical Pre-Execution Checks

**Add these checks BEFORE starting Increment 1**:

```python
# Pre-execution validation script
def validate_environment():
    checks = {
        "database_exists": os.path.exists("data/02_stg/stg_disaster_response.db"),
        "venv_active": sys.prefix != sys.base_prefix,
        "python_version": sys.version_info >= (3, 12),
        "disk_space_gb": shutil.disk_usage(".").free / (1024**3) > 10,
        "memory_available_gb": psutil.virtual_memory().available / (1024**3) > 8,
        "production_model_exists": os.path.exists("model/disaster_rf_v25-09-16_prod_2025-09-19.pkl"),
        "required_scripts": all([
            os.path.exists("scripts/03_create_experimental_model.py"),
            os.path.exists("scripts/compare_models.py"),
            os.path.exists("src/disasterproject/models/pipeline.py")
        ]),
        "required_packages": all([
            importlib.util.find_spec(pkg) for pkg in [
                "sklearn", "pandas", "numpy", "joblib", "sqlalchemy"
            ]
        ])
    }
    
    failed_checks = [k for k, v in checks.items() if not v]
    if failed_checks:
        raise EnvironmentError(f"Pre-execution checks failed: {failed_checks}")
    return checks
```

## 📊 Logical Gaps & Missing Dependencies

### Gap 1: Undefined Functions
- **Missing**: No verification that `get_multilabel_class_weights()` exists
- **Solution**: Add explicit import check and fallback implementation

### Gap 2: Comparison Baseline
- **Missing**: No check if production model can be loaded for comparison
- **Solution**: Add baseline validation before starting increments

### Gap 3: Eval Set Dependencies
- **Missing**: Plan says "acceptable" if eval_ids missing, but inconsistent evaluation invalidates comparisons
- **Solution**: REQUIRE consistent eval set - create if missing:

```python
# Add to pre-execution setup
if not os.path.exists("data/04_fct/eval_ids.csv"):
    print("Creating consistent eval set...")
    # Generate and save eval_ids with fixed seed
    create_eval_split(test_size=0.2, random_state=42)
```

## ⚠️ Risky Assumptions

### Assumption 1: Training Time Estimates
- **Risk**: "2-4 hours" for class weighting could be 6+ hours if pipeline recreation needed
- **Mitigation**: Add timeout of 2 hours per increment with auto-skip

### Assumption 2: Memory Requirements
- **Risk**: RandomForest with class weights might need 2x memory
- **Mitigation**: Monitor memory usage, implement chunked training if needed

### Assumption 3: LogisticRegression Convergence
- **Risk**: `max_iter=1000` might not converge with imbalanced data
- **Mitigation**: Use `LogisticRegressionCV` with automatic iteration selection:

```python
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(
    cv=3, 
    max_iter=10000,  # Higher limit
    solver='saga',    # Better for multiclass
    penalty='l1',     # Sparsity for feature selection
    random_state=42,
    n_jobs=-1
)
```

## 🔄 Better Sequencing & Increment Breakdown

### Recommended Sequence (REVISED):

```
0. PRE-EXECUTION VALIDATION (30 min)
   ├── Environment checks (automated)
   ├── Create eval split if missing
   ├── Backup current production model
   ├── Test load production baseline
   └── Create experiment directory structure

1. INFRASTRUCTURE SETUP (30 min)
   ├── Modify 03_create_experimental_model.py (add --algorithm)
   ├── Implement CustomMultiOutputClassifier
   ├── Add class weight calculation utilities
   └── Verify all imports and functions work

2. INCREMENT 1: LogisticRegression WITHOUT Weights (1 hour)
   ├── Establish LR baseline performance
   ├── Verify LR works with pipeline
   ├── Compare to RF baseline
   └── STOP if F1 < 0.85 (too poor to continue)

3. INCREMENT 2: LogisticRegression WITH Weights (2 hours)
   ├── Apply per-label class weights
   ├── Handle zero-positive labels properly
   ├── Compare to LR baseline (not RF)
   └── STOP if no recall improvement

4. INCREMENT 3: RandomForest WITH Weights (2 hours) [OPTIONAL]
   ├── Only if LR+weights fails gates
   ├── Memory-intensive, run last
   └── May timeout - acceptable

5. INCREMENT 4: Threshold Optimization (1 hour)
   ├── Apply to best model from 2 or 3
   ├── Validate on held-out set
   └── Save threshold config
```

## 🛑 Stop & Ask Human Triggers

### Add these automatic stop conditions:

```python
STOP_CONDITIONS = {
    "catastrophic_f1_drop": lambda metrics: metrics['f1_weighted'] < 0.80,
    "no_recall_improvement": lambda metrics: metrics['critical_recall_delta'] <= 0,
    "memory_exceeded": lambda: psutil.virtual_memory().percent > 90,
    "time_exceeded": lambda elapsed: elapsed > 3600 * 2,  # 2 hours per increment
    "training_failed": lambda exception: isinstance(exception, (MemoryError, ValueError)),
    "model_too_large": lambda size_mb: size_mb > 1000,
    "convergence_failed": lambda model: hasattr(model, 'n_iter_') and model.n_iter_ == model.max_iter
}

def should_stop_execution(metrics, exception=None, elapsed=0):
    for condition_name, check in STOP_CONDITIONS.items():
        if check(metrics):  # Adjust parameters based on condition
            return True, f"Stop condition triggered: {condition_name}"
    return False, None
```

## 🔧 Technical Clarifications Needed

### 1. Class Weight Implementation
**Problem**: Setting weights after fitting doesn't work.

**Correct Approach**:
```python
from sklearn.base import clone
from sklearn.multioutput import MultiOutputClassifier

class WeightedMultiOutputClassifier(MultiOutputClassifier):
    def __init__(self, estimator, class_weights_list=None, n_jobs=None):
        super().__init__(estimator, n_jobs=n_jobs)
        self.class_weights_list = class_weights_list
    
    def fit(self, X, y, sample_weight=None):
        y = self._validate_data(X='no_validation', y=y, multi_output=True)
        
        if self.class_weights_list is None:
            return super().fit(X, y, sample_weight)
        
        # Create individual estimators with proper weights
        self.estimators_ = []
        for i, column in enumerate(y.T):
            estimator = clone(self.estimator)
            if i < len(self.class_weights_list):
                if hasattr(estimator, 'class_weight'):
                    estimator.class_weight = self.class_weights_list[i]
            
            # Handle zero-positive case
            if len(np.unique(column)) == 1:
                # Create dummy predictor for single-class labels
                estimator = DummyClassifier(strategy='constant', constant=0)
            
            estimator.fit(X, column)
            self.estimators_.append(estimator)
        
        return self
```

### 2. Zero-Positive Label Handling
**Better Approach**: Use DummyClassifier for zero-positive labels instead of arbitrary weights:

```python
def handle_zero_positive_labels(y_train, label_names):
    """Identify labels with no positive examples"""
    zero_positive_mask = (y_train.sum(axis=0) == 0)
    zero_positive_labels = [label_names[i] for i, is_zero in enumerate(zero_positive_mask) if is_zero]
    
    if zero_positive_labels:
        print(f"WARNING: Labels with no positive examples: {zero_positive_labels}")
        print("These will use DummyClassifier(constant=0)")
    
    return zero_positive_mask
```

## 📝 Error Recovery Paths

### Recovery Strategy Matrix:

| Error Type | Detection | Recovery | Continue? |
|------------|-----------|----------|-----------|
| OOM during training | MemoryError exception | Reduce batch size, use SGD | No - Stop |
| Convergence failure | n_iter == max_iter | Increase max_iter, change solver | Yes - Log |
| Database missing | FileNotFoundError | Recreate from raw data | No - Stop |
| Model corruption | joblib.load fails | Use previous checkpoint | Yes - Retry |
| Comparison script fails | No metrics output | Compute metrics manually | Yes - Continue |
| Class weight calc fails | ValueError | Use uniform weights | Yes - Document |
| Disk space exhausted | OSError | Clean experiments/ | No - Stop |
| Import errors | ModuleNotFoundError | pip install missing | No - Stop |

### Automated Recovery Implementation:

```python
def safe_increment_execution(increment_func, increment_name, max_retries=2):
    """Execute increment with automatic recovery"""
    for attempt in range(max_retries):
        try:
            # Create checkpoint before execution
            checkpoint_dir = f"experiments/checkpoints/{increment_name}_{attempt}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Execute with monitoring
            with ResourceMonitor(max_memory_gb=8, max_time_hours=2):
                result = increment_func()
            
            return result
            
        except MemoryError:
            if attempt == 0:
                # Try with reduced parameters
                reduce_memory_footprint()
                continue
            else:
                log_and_stop(f"MemoryError on attempt {attempt+1}")
                break
                
        except ConvergenceWarning as e:
            log_warning(f"Convergence issue: {e}")
            # Continue but note in report
            return {"status": "partial", "warning": str(e)}
            
        except Exception as e:
            if attempt < max_retries - 1:
                log_warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(30)  # Brief pause
                continue
            else:
                log_and_stop(f"Final attempt failed: {e}")
                break
    
    return {"status": "failed", "attempts": max_retries}
```

## 🎯 Revised Success Criteria

### Per-Increment Gates (STRICT):

**Increment 1 (LR Baseline)**:
- ✅ Model trains successfully
- ✅ F1-weighted ≥ 0.85 (minimum viable)
- ✅ Model size < 10MB
- ⚠️ Document any convergence warnings

**Increment 2 (LR + Weights)**:
- ✅ Critical recall improvement ≥ 10% over LR baseline
- ✅ F1-weighted ≥ 0.90
- ✅ At least 3 critical categories > 50% recall
- 🛑 STOP if recall doesn't improve

**Increment 3 (RF + Weights - Optional)**:
- ⚠️ Only run if LR+weights fails
- ✅ Complete within 2 hours or skip
- ✅ Memory usage < 90%

**Increment 4 (Thresholds)**:
- ✅ All critical categories ≥ 60% recall
- ✅ F1-weighted drop ≤ 5% from best model
- ✅ Threshold config saved and validated

## 📋 Pre-Execution Checklist (REVISED)

```bash
# Run this validation script before starting
python scripts/validate_ml_execution_environment.py

✓ Environment validated
✓ Database exists: data/02_stg/stg_disaster_response.db
✓ Production baseline loadable
✓ Eval split consistent
✓ 15GB disk space available  
✓ 8GB memory available
✓ Virtual environment active
✓ All dependencies installed
✓ Backup created: model/backups/2025-11-04/
✓ Experiment directory ready: experiments/experimental_runs/2025-11-04/
```

## 🚀 Final Recommendations

1. **Change execution order**: LR first (proven), then add complexity
2. **Fix class weight implementation**: Use custom MultiOutputClassifier  
3. **Add strict stop conditions**: Prevent wasted computation
4. **Require consistent eval set**: No "acceptable" randomness
5. **Add resource monitoring**: Prevent OOM and timeout
6. **Document partial successes**: Some improvement > no improvement
7. **Create restore points**: Enable safe rollback

## Next Steps

1. Create `scripts/validate_ml_execution_environment.py` with all checks
2. Implement `WeightedMultiOutputClassifier` in pipeline.py
3. Add resource monitoring wrapper
4. Update execution plan with these recommendations
5. Test pre-execution validation locally before overnight run
