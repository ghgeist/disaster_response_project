---
title: "Update ML Models"
date: "2025-09-02"
status: "active"
tags: ["documentation", "instruction"]
author: "runner"
related: []
---


## 🚀 **Update ML Models**

### **Phase 1: Automated Experiment Execution (Currently Running!)**
```bash

python run_all_experiments.py
```

**What's happening right now:**
- 🧪 **baseline_no_sampling** - Currently running
- ⏳ **smote_conservative** - Next in queue
- ⏳ **adasyn_moderate** - Third in queue  
- ⏳ **conservative_sampling** - Final experiment

**Timeline:** ~20 minutes total (perfect for your workout!)
# ✅ Complete!


### **Phase 2: Post-Workout Analysis (15-20 minutes)**

#### **Step 1: Check Experiment Status**
```bash
# Check if all experiments completed
ls -la experiments/
ls -la models/*.pkl

# View the automated report
cat experiment_report_*.json
```

#### **Step 2: Compare All Results**
```bash
# Run comprehensive model comparison
python scripts/compare_models.py
```

#### **Step 3: Analyze Performance Metrics**
```bash
# Check detailed results
ls -la data/04_fct/
cat data/04_fct/fct_*_prediction_results.csv
```

---

### **Phase 3: Model Selection & Validation (10-15 minutes)**

#### **Step 1: Identify Best Performing Model**
Based on the legacy analysis, focus on:
- **Recall for positive cases** (target: >25%)
- **F1-score for positive cases** (target: >20%)
- **Overall macro average** (target: >60%)

#### **Step 2: Test Best Model in Web Application**
```bash
# Update app config to use best model
# Edit app/config.py to point to best model

# Test the web application
cd app && python app.py
# Test predictions at http://127.0.0.1:3000
```

---

### **Phase 4: Advanced Experimentation (Optional - 30-45 minutes)**

#### **If Results Need Improvement:**
```bash
# Run systematic testing framework
python scripts/systematic_testing_framework.py

# Test custom parameters
python scripts/train_model.py data/02_stg/stg_disaster_response.db models/custom_model.pkl
# Select option 5 (Custom experiment)
```

#### **Hyperparameter Tuning:**
- Modify `src/disaster_classifier/models/pipeline.py`
- Test different n-gram ranges: (1,1), (1,2), (2,2)
- Adjust RandomForest parameters
- Test different sampling ratios

---

## �� **Expected Results Based on Legacy Analysis**

### **Current Baseline (from legacy README):**
- **Recall for positive cases**: 4-8% ❌
- **F1-score for positive cases**: 7-14% ❌
- **Training time**: 7 minutes ⚠️
- **Model size**: 561MB ⚠️

### **Expected Improvements with Sampling:**
- **SMOTE/ADASYN**: Should boost recall to 20-30% ✅
- **Conservative sampling**: Should balance precision/recall ✅
- **Training time**: Should remain <5 minutes ✅
- **Model size**: Should be manageable ✅

---

## �� **Success Criteria for Tomorrow**

### **Minimum Acceptable Results:**
- ✅ **Recall >15%** for positive cases (4x improvement)
- ✅ **F1-score >12%** for positive cases (2x improvement)
- ✅ **Training time <5 minutes** (faster than legacy)
- ✅ **Web app functional** with best model

### **Excellent Results:**
- 🎯 **Recall >25%** for positive cases (6x improvement)
- 🎯 **F1-score >20%** for positive cases (3x improvement)
- �� **Training time <3 minutes** (2x faster)
- 🎯 **All 36 categories** showing improved performance

---

## 💡 **Pro Tips for Analysis**

### **Key Metrics to Focus On:**
1. **Recall for class '1'** (positive cases) - This was the main problem
2. **Macro average F1-score** - Overall model performance
3. **Categories with <10% recall** - Identify problem areas
4. **Training time** - Efficiency matters

### **Red Flags to Watch For:**
- ❌ Recall still <10% after sampling
- ❌ Precision drops below 70%
- ❌ Training time >10 minutes
- ❌ Model size >1GB

### **Success Indicators:**
- ✅ Recall >20% for positive cases
- ✅ F1-score >15% for positive cases
- ✅ Training time <5 minutes
- ✅ Web app predictions make sense

---

## 🏃‍♂️ **Your Workout Timeline**

**Right Now (0-20 minutes):** Experiments running automatically
**After Workout (20-40 minutes):** Analysis and model selection
**Optional (40-60 minutes):** Advanced tuning if needed

**Perfect timing!** 💪 Your experiments will be done when you get back, and you'll have a clear path to dramatically improve your model performance compared to the legacy baseline.