#!/usr/bin/env python3
"""
Quick diagnostic to compare child_alone behavior between production and experimental models.
"""

import sys
import os
sys.path.append('src')

import joblib
from disasterproject.utils.config import TARGET_COLUMNS

def test_child_alone_classifier(model_path, model_name):
    """Test the child_alone classifier behavior."""

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    print(f"\n🔍 Testing {model_name}")
    print("=" * 50)

    model = joblib.load(model_path)

    # Test with a simple message that clearly has nothing to do with children
    test_msg = "The weather is nice today"
    pred = model.predict([test_msg])[0]
    probs = model.predict_proba([test_msg])

    child_alone_idx = TARGET_COLUMNS.index('child_alone')
    child_alone_pred = pred[child_alone_idx]
    child_alone_prob_array = probs[child_alone_idx][0]

    print(f"Message: '{test_msg}'")
    print(f"child_alone prediction: {child_alone_pred}")
    print(f"child_alone prob array shape: {child_alone_prob_array.shape}")
    print(f"child_alone prob array: {child_alone_prob_array}")

    # Check for degenerate classifier
    if child_alone_prob_array.shape[0] == 1:
        print("⚠️  WARNING: Degenerate classifier (only one class)")
        print("   This suggests no positive examples in training data")
    elif child_alone_prob_array.shape[0] == 2:
        print(f"✅ Normal binary classifier")
        print(f"   Prob of negative class: {child_alone_prob_array[0]:.3f}")
        print(f"   Prob of positive class: {child_alone_prob_array[1]:.3f}")

    # Test with a message that mentions children
    child_msg = "Children need help after the disaster"
    child_pred = model.predict([child_msg])[0]
    child_probs = model.predict_proba([child_msg])
    child_alone_pred2 = child_pred[child_alone_idx]
    child_alone_prob_array2 = child_probs[child_alone_idx][0]

    print(f"\nChild-related message: '{child_msg}'")
    print(f"child_alone prediction: {child_alone_pred2}")
    if child_alone_prob_array2.shape[0] == 2:
        print(f"child_alone probability: {child_alone_prob_array2[1]:.3f}")
    else:
        print(f"child_alone prob array: {child_alone_prob_array2}")

def main():
    """Compare child_alone behavior between models."""

    production_model = "model/disaster_rf_v1-2-0_prod_2025-09-11.pkl"
    experimental_model = "experiments/results/2025-09-16-comprehensive-grid-search-optimized-model.pkl"

    test_child_alone_classifier(production_model, "PRODUCTION MODEL")
    test_child_alone_classifier(experimental_model, "EXPERIMENTAL MODEL")

    print(f"\n" + "=" * 60)
    print("🎯 ANALYSIS:")
    print("If both models have the same issue, it's a training data problem.")
    print("If only experimental has the issue, it's a hyperparameter problem.")

if __name__ == "__main__":
    main()