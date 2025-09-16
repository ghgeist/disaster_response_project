#!/usr/bin/env python3
"""
Compare confidence scores between production and experimental disaster response models.

Usage: python test_experimental_model.py
"""

import sys
import os
sys.path.append('src')

import joblib
import numpy as np
from disasterproject.utils.config import TARGET_COLUMNS

def load_models():
    """Load both production and experimental models."""
    production_path = 'model/disaster_rf_v1-2-0_prod_2025-09-11.pkl'
    experimental_path = 'experiments/results/2025-09-16-comprehensive-grid-search-optimized-model.pkl'

    models = {}

    if os.path.exists(production_path):
        print("🏭 Loading production model...")
        models['production'] = joblib.load(production_path)
        print("✅ Production model loaded!")
    else:
        print(f"❌ Production model not found: {production_path}")
        models['production'] = None

    if os.path.exists(experimental_path):
        print("🧪 Loading experimental model...")
        models['experimental'] = joblib.load(experimental_path)
        print("✅ Experimental model loaded!")
    else:
        print(f"❌ Experimental model not found: {experimental_path}")
        models['experimental'] = None

    return models

def get_model_predictions(model, message):
    """Get predictions and probabilities from a model, handling degenerate classifiers."""
    if model is None:
        return None, None, None

    predictions = model.predict([message])[0]
    probabilities = model.predict_proba([message])

    # Extract positive class probabilities, handling degenerate classifiers
    pos_probs = []
    for prob_array in probabilities:
        if prob_array.shape[1] == 2:  # Normal binary classifier
            pos_probs.append(prob_array[0][1])
        else:  # Degenerate classifier (only one class)
            pos_probs.append(0.0 if prob_array[0][0] == 1.0 else prob_array[0][0])

    # Get predicted categories (where prediction = 1)
    predicted_categories = []
    for j, (pred, prob) in enumerate(zip(predictions, pos_probs)):
        if pred == 1:
            predicted_categories.append((TARGET_COLUMNS[j], prob))

    return predictions, pos_probs, predicted_categories

def compare_models():
    """Compare confidence scores between production and experimental models."""

    # Load both models
    models = load_models()

    if models['production'] is None and models['experimental'] is None:
        print("❌ No models available for comparison!")
        return

    print(f"📊 Models predict {len(TARGET_COLUMNS)} categories")
    print()

    # Test messages covering different disaster scenarios
    test_messages = [
        "We need medical help urgently, people are injured in the earthquake",
        "The storm destroyed our water supply, we need clean water immediately",
        "Our shelter collapsed, need emergency housing for 50 families",
        "Please send food supplies, we have been without food for 3 days",
        "The bridge is destroyed, we cannot get to the hospital",
        "Weather report: Hurricane approaching with 120 mph winds",
        "Security needed at the refugee camp, situation getting tense",
        "Children are missing after the flood, please help with search and rescue"
    ]

    print("🔍 CONFIDENCE SCORE COMPARISON")
    print("=" * 100)

    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. MESSAGE: {message}")
        print("-" * 80)

        # Get predictions from both models
        prod_predictions, prod_probs, prod_categories = get_model_predictions(models['production'], message)
        exp_predictions, exp_probs, exp_categories = get_model_predictions(models['experimental'], message)

        # Show predicted categories side by side
        print("📋 PREDICTED CATEGORIES:")

        if models['production'] is not None:
            print(f"   🏭 PRODUCTION ({len(prod_categories)} categories):")
            for category, confidence in prod_categories:
                print(f"      ✅ {category:<25} {confidence:.3f}")

        if models['experimental'] is not None:
            print(f"   🧪 EXPERIMENTAL ({len(exp_categories)} categories):")
            for category, confidence in exp_categories:
                print(f"      ✅ {category:<25} {confidence:.3f}")

        # Compare top probabilities side by side
        print("\n🔝 TOP 5 CONFIDENCE SCORES COMPARISON:")
        print(f"{'CATEGORY':<25} {'PRODUCTION':<12} {'EXPERIMENTAL':<12} {'DIFFERENCE':<12}")
        print("-" * 65)

        if prod_probs is not None and exp_probs is not None:
            # Get top categories by experimental model confidence
            exp_prob_pairs = list(zip(TARGET_COLUMNS, exp_probs))
            exp_prob_pairs.sort(key=lambda x: x[1], reverse=True)

            for category, exp_prob in exp_prob_pairs[:5]:
                idx = TARGET_COLUMNS.index(category)
                prod_prob = prod_probs[idx] if prod_probs else 0.0

                diff = exp_prob - prod_prob
                diff_str = f"{diff:+.3f}"

                # Color coding for differences
                if diff > 0.05:
                    diff_indicator = "📈"
                elif diff < -0.05:
                    diff_indicator = "📉"
                else:
                    diff_indicator = "➡️"

                print(f"{category:<25} {prod_prob:<12.3f} {exp_prob:<12.3f} {diff_str:<12} {diff_indicator}")

        elif prod_probs is not None:
            print("   (Only production model available)")
            prod_prob_pairs = list(zip(TARGET_COLUMNS, prod_probs))
            prod_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            for category, prob in prod_prob_pairs[:5]:
                print(f"{category:<25} {prob:.3f}")

        elif exp_probs is not None:
            print("   (Only experimental model available)")
            exp_prob_pairs = list(zip(TARGET_COLUMNS, exp_probs))
            exp_prob_pairs.sort(key=lambda x: x[1], reverse=True)
            for category, prob in exp_prob_pairs[:5]:
                print(f"{category:<25} {prob:.3f}")

if __name__ == "__main__":
    compare_models()