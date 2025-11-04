#!/usr/bin/env python3
"""
Diagnostic script to check why the starvation message isn't being categorized.
"""

import sys
from pathlib import Path
import os

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

# Set up environment for app imports
os.environ.setdefault('FLASK_ENV', 'development')

from app.services import ModelService
from disasterproject.utils.config import (
    TAXONOMY,
    CRITICAL_LABELS,
    EXCLUDE_FROM_CONSTRAINTS,
    HIERARCHY_CRITICAL_THRESHOLD_REDUCTION
)
from disasterproject.hierarchy import apply_hierarchy

def diagnose_message(message: str):
    """Diagnose why a message isn't being categorized."""
    
    print(f"\n{'='*80}")
    print(f"DIAGNOSING MESSAGE:")
    print(f"'{message}'")
    print(f"{'='*80}\n")
    
    # Initialize model service
    model_path = project_root / "model" / "disaster_rf_v1-2-0_prod_2025-09-11.pkl"
    if not model_path.exists():
        # Try to find any model
        model_dir = project_root / "model"
        models = list(model_dir.glob("*.pkl"))
        if models:
            model_path = models[0]
            print(f"⚠️  Using model: {model_path.name}")
        else:
            print(f"❌ No model found in {model_dir}")
            return
    
    model_service = ModelService(model_path)
    model_service.load_model()
    
    # Get raw predictions
    prediction = model_service.predict(message)
    raw_labels = prediction.get('labels', {})
    raw_probabilities = prediction.get('probabilities', {})
    
    print("📊 RAW MODEL OUTPUT:")
    print("-" * 80)
    
    # Show relevant categories
    relevant_categories = ['related', 'food', 'medical_help', 'request', 'aid_related', 'child_alone']
    
    print("\n🔍 RELEVANT CATEGORIES:")
    for category in relevant_categories:
        if category in raw_probabilities:
            prob = raw_probabilities[category]
            label = raw_labels.get(category, 0)
            threshold = 0.5  # Default threshold
            status = "✅ ABOVE" if prob >= threshold else "❌ BELOW"
            print(f"  {category:20s} | Prob: {prob:.4f} | Label: {label} | Threshold: {threshold:.2f} | {status}")
    
    # Show all probabilities sorted
    print("\n📈 ALL PROBABILITIES (sorted by value):")
    sorted_probs = sorted(raw_probabilities.items(), key=lambda x: x[1], reverse=True)
    for category, prob in sorted_probs[:10]:  # Top 10
        label = raw_labels.get(category, 0)
        print(f"  {category:25s} | {prob:.4f} | Label: {label}")
    
    # Check thresholds
    print("\n🎯 THRESHOLDS:")
    thresholds = model_service._get_thresholds_map()
    print(f"  Using model service thresholds: {len(thresholds)} thresholds loaded")
    for category in relevant_categories:
        if category in thresholds:
            print(f"  {category:20s} | Threshold: {thresholds[category]:.4f}")
    
    # Apply hierarchy processing (like the route does)
    print("\n🔄 APPLYING HIERARCHY PROCESSING:")
    print("-" * 80)
    
    # Use default thresholds (0.5) like the route does
    route_thresholds = {label: 0.5 for label in raw_probabilities.keys()}
    print(f"  Using route thresholds (0.5 for all): {len(route_thresholds)} labels")
    
    fixed_probabilities, fixed_labels = apply_hierarchy(
        probs=raw_probabilities,
        thresholds=route_thresholds,
        taxonomy=TAXONOMY,
        critical_labels=CRITICAL_LABELS,
        exclude=EXCLUDE_FROM_CONSTRAINTS,
        critical_threshold_reduction=HIERARCHY_CRITICAL_THRESHOLD_REDUCTION
    )
    
    print(f"\n  Critical threshold reduction: {HIERARCHY_CRITICAL_THRESHOLD_REDUCTION}")
    print(f"  Critical labels: {sorted(CRITICAL_LABELS)}")
    
    print("\n📊 AFTER HIERARCHY PROCESSING:")
    for category in relevant_categories:
        if category in fixed_probabilities:
            raw_prob = raw_probabilities[category]
            fixed_prob = fixed_probabilities[category]
            raw_label = raw_labels.get(category, 0)
            fixed_label = fixed_labels.get(category, 0)
            
            prob_change = fixed_prob - raw_prob
            label_change = fixed_label - raw_label
            
            prob_str = f"{raw_prob:.4f} → {fixed_prob:.4f}"
            label_str = f"{raw_label} → {fixed_label}"
            
            if prob_change != 0 or label_change != 0:
                print(f"  {category:20s} | Prob: {prob_str:20s} | Label: {label_str}")
            else:
                print(f"  {category:20s} | Prob: {raw_prob:.4f} (unchanged) | Label: {raw_label} (unchanged)")
    
    # Check what would be displayed
    print("\n🎨 DISPLAY RESULTS:")
    print("-" * 80)
    
    fixed_predictions = []
    for category, label in fixed_labels.items():
        if label == 1 and category != 'related':
            fixed_predictions.append({
                "category": category,
                "confidence": fixed_probabilities.get(category, 0.0)
            })
    
    if fixed_predictions:
        print(f"  ✅ {len(fixed_predictions)} categories would be displayed:")
        for pred in sorted(fixed_predictions, key=lambda p: p['confidence'], reverse=True):
            print(f"    - {pred['category']:20s} ({pred['confidence']:.4f})")
    else:
        related_prob = fixed_probabilities.get('related', 0)
        if related_prob > 0.5:
            print(f"  ⚠️  No categories displayed (but related={related_prob:.4f} > 0.5)")
            print(f"     This triggers: 'This message is disaster-related, but doesn't match our specific emergency categories.'")
        else:
            print(f"  ❌ No categories displayed (related={related_prob:.4f} <= 0.5)")
            print(f"     This triggers: 'This message doesn't appear to be about a disaster or emergency.'")
    
    # Check if using model service thresholds would help
    print("\n💡 USING MODEL SERVICE THRESHOLDS INSTEAD:")
    print("-" * 80)
    
    service_thresholds = model_service._get_thresholds_map()
    # Fill in missing thresholds with 0.5
    complete_thresholds = {label: service_thresholds.get(label, 0.5) for label in raw_probabilities.keys()}
    
    fixed_probs_v2, fixed_labels_v2 = apply_hierarchy(
        probs=raw_probabilities,
        thresholds=complete_thresholds,
        taxonomy=TAXONOMY,
        critical_labels=CRITICAL_LABELS,
        exclude=EXCLUDE_FROM_CONSTRAINTS,
        critical_threshold_reduction=HIERARCHY_CRITICAL_THRESHOLD_REDUCTION
    )
    
    fixed_predictions_v2 = []
    for category, label in fixed_labels_v2.items():
        if label == 1 and category != 'related':
            fixed_predictions_v2.append({
                "category": category,
                "confidence": fixed_probs_v2.get(category, 0.0)
            })
    
    if fixed_predictions_v2:
        print(f"  ✅ {len(fixed_predictions_v2)} categories would be displayed:")
        for pred in sorted(fixed_predictions_v2, key=lambda p: p['confidence'], reverse=True):
            print(f"    - {pred['category']:20s} ({pred['confidence']:.4f})")
    else:
        related_prob = fixed_probs_v2.get('related', 0)
        print(f"  ⚠️  Still no categories (related={related_prob:.4f})")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    message = "My child is dying of starvation, I have received nothing"
    diagnose_message(message)

