#!/usr/bin/env python3
"""
Quick test script for the experimental disaster response model.

Usage: python test_experimental_model.py
"""

import sys
import os
sys.path.append('src')

import joblib
import numpy as np
from disasterproject.utils.config import TARGET_COLUMNS

def test_model():
    """Test the experimental model with sample messages."""

    # Load the experimental model
    model_path = 'experiments/results/2025-09-16-comprehensive-grid-search-optimized-model.pkl'

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    print("🔍 Loading experimental model...")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
    print(f"📊 Model predicts {len(TARGET_COLUMNS)} categories")
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

    print("🧪 Testing model with sample disaster messages:")
    print("=" * 80)

    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. MESSAGE: {message}")
        print("-" * 60)

        # Get predictions (0 or 1 for each category)
        predictions = model.predict([message])[0]

        # Get probabilities for each category
        try:
            probabilities = model.predict_proba([message])
            # Extract probability of positive class (class 1) for each category
            pos_probs = [probs[1] if len(probs[0]) > 1 else probs[0][0] for probs in probabilities]
        except:
            pos_probs = [0.5] * len(predictions)

        # Show predicted categories (where prediction = 1)
        predicted_categories = []
        for j, (pred, prob) in enumerate(zip(predictions, pos_probs)):
            if pred == 1:
                predicted_categories.append((TARGET_COLUMNS[j], prob))

        if predicted_categories:
            print("📋 PREDICTED CATEGORIES:")
            for category, confidence in predicted_categories:
                print(f"   ✅ {category:<25} (confidence: {confidence:.3f})")
        else:
            print("📋 No categories predicted (all probabilities below threshold)")

        # Show top 3 probabilities regardless of threshold
        prob_pairs = list(zip(TARGET_COLUMNS, pos_probs))
        prob_pairs.sort(key=lambda x: x[1], reverse=True)

        print("🔝 TOP 3 PROBABILITIES:")
        for category, prob in prob_pairs[:3]:
            print(f"   📊 {category:<25} {prob:.3f}")

if __name__ == "__main__":
    test_model()