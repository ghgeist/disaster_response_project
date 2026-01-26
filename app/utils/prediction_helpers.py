"""
Prediction formatting and processing helpers.
"""
from typing import Any, Dict, List, Mapping

from app.utils.validation import sanitize_input, validate_message_input


def synthesize_probabilities(labels_dict: Mapping[str, int] | None) -> Dict[str, float]:
    """
    Build a probability map from binary labels when predict_proba is unavailable.

    Args:
        labels_dict: Mapping of label names to 0/1 predictions.

    Returns:
        Dict of label names to float probabilities.
    """
    if not labels_dict:
        return {}
    return {label: 1.0 if value == 1 else 0.0 for label, value in labels_dict.items()}


def format_predictions_for_display(
    prediction_result: Mapping[str, Any] | None,
    exclude_related: bool = True,
) -> List[Dict[str, float]]:
    """
    Convert prediction output into a display-friendly list of categories.

    Args:
        prediction_result: Dict containing "labels" and optional "probabilities".
        exclude_related: When True, omit the "related" meta-category.

    Returns:
        Sorted list of dicts with category and confidence keys.
    """
    if not prediction_result:
        return []

    labels = prediction_result.get("labels", {}) or {}
    probabilities = prediction_result.get("probabilities", {}) or {}

    if not probabilities and labels:
        probabilities = synthesize_probabilities(labels)

    predictions: List[Dict[str, float]] = []
    for category, label in labels.items():
        if label == 1 and (not exclude_related or category != "related"):
            predictions.append(
                {
                    "category": category,
                    "confidence": probabilities.get(category, 0.0),
                }
            )

    return sorted(predictions, key=lambda item: item["confidence"], reverse=True)


def process_prediction_result(
    model_service: Any,
    query: str,
    sanitize: bool = True,
) -> Dict[str, Any]:
    """
    Sanitize, validate, predict, and format a user message.

    Args:
        model_service: ModelService-like instance with a predict() method.
        query: Raw user input text.
        sanitize: When True, sanitize the input before validation/prediction.

    Returns:
        Dict containing validation status, query, labels, probabilities, and display data.
    """
    cleaned_query = sanitize_input(query) if sanitize else query
    is_valid, error_message = validate_message_input(cleaned_query)

    result: Dict[str, Any] = {
        "query": cleaned_query,
        "is_valid": is_valid,
        "error_message": error_message,
        "labels": {},
        "probabilities": {},
        "sorted_predictions": [],
    }

    if not is_valid:
        return result

    prediction = model_service.predict(cleaned_query)
    labels = prediction.get("labels", {}) or {}
    probabilities = prediction.get("probabilities", {}) or {}

    if not probabilities and labels:
        probabilities = synthesize_probabilities(labels)

    prediction_payload = {
        "labels": labels,
        "probabilities": probabilities,
    }
    sorted_predictions = format_predictions_for_display(prediction_payload)

    result.update(
        {
            "prediction": prediction_payload,
            "labels": labels,
            "probabilities": probabilities,
            "sorted_predictions": sorted_predictions,
        }
    )

    return result
