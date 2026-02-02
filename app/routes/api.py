"""
Storm Signal dashboard API: feed, metrics, categories, and classification.
"""
import hashlib
import logging
import math
import random
from datetime import datetime, timedelta, timezone

from flask import Blueprint, current_app, jsonify, request, send_from_directory

from app.extensions import csrf
from app.services.errors import DataServiceError
from app.services.model_service import ModelServiceError
from app.utils.formatting import format_request_context
from app.utils.prediction_helpers import process_prediction_result

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")
csrf.exempt(api_bp)

CATEGORY_DISPLAY_NAMES = {
    "search_and_rescue": "Search & Rescue",
    "infrastructure_related": "Infrastructure",
    "aid_centers": "Aid Centers",
    "other_infrastructure": "Other Infrastructure",
    "weather_related": "Weather Related",
    "direct_report": "Direct Report",
    "child_alone": "Child Alone",
    "medical_products": "Medical Products",
    "other_aid": "Other Aid",
    "other_weather": "Other Weather",
}

CATEGORY_GROUPS = {
    "Critical Needs": [
        "Medical Help",
        "Medical Products",
        "Search & Rescue",
        "Water",
        "Food",
        "Shelter",
        "Security",
        "Hospitals",
    ],
    "Infrastructure": [
        "Transport",
        "Buildings",
        "Electricity",
        "Tools",
        "Shops",
        "Aid Centers",
        "Other Infrastructure",
    ],
    "Weather": [
        "Floods",
        "Storm",
        "Fire",
        "Earthquake",
        "Cold",
        "Other Weather",
    ],
    "Other": [
        "Missing People",
        "Refugees",
        "Death",
        "Clothing",
        "Money",
        "Other Aid",
        "Military",
        "Child Alone",
        "Request",
        "Offer",
        "Direct Report",
    ],
}

CRITICAL_INTERNAL_CATEGORIES = {
    "medical_help",
    "medical_products",
    "search_and_rescue",
    "water",
    "food",
    "shelter",
    "security",
    "hospitals",
}


def to_display_name(internal: str) -> str:
    """Convert internal category names to display names."""
    return CATEGORY_DISPLAY_NAMES.get(internal, internal.replace("_", " ").title())


def _safe_category_display(internal) -> str:
    """Return display name for a category key; handle None/NaN keys to avoid JSON/crash."""
    if internal is None:
        logger.debug("Category key was None; coercing to Unknown (upstream data check).")
        return "Unknown"
    if isinstance(internal, float) and math.isnan(internal):
        logger.debug("Category key was NaN; coercing to Unknown (upstream data check).")
        return "Unknown"
    if isinstance(internal, str) and internal:
        return to_display_name(internal)
    logger.debug(
        "Category key was non-string (%s); coercing to str (upstream data check).",
        type(internal).__name__,
    )
    return str(internal) if internal is not None else "Unknown"


def calculate_severity(probabilities: dict) -> str:
    """Determine severity based on critical category probabilities."""
    critical_count = sum(
        1
        for category, probability in probabilities.items()
        if category in CRITICAL_INTERNAL_CATEGORIES and _safe_float_prob(probability) > 0.5
    )
    critical_probabilities = [
        _safe_float_prob(probability)
        for category, probability in probabilities.items()
        if category in CRITICAL_INTERNAL_CATEGORIES
    ]
    max_confidence = max(critical_probabilities) if critical_probabilities else 0.0
    if critical_count >= 2 or max_confidence > 0.85:
        return "HIGH"
    if critical_count >= 1 or max_confidence > 0.70:
        return "MEDIUM"
    return "LOW"


def generate_timestamp_for_id(raw_id) -> datetime:
    """Return a stable timestamp for a record based on its id (last 6 hours, deterministic)."""
    key = str(raw_id).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    fraction = int(digest[:12], 16) / (16**12)
    hours_ago = fraction * 6
    return datetime.now(timezone.utc) - timedelta(hours=hours_ago)


GENRE_TO_SOURCE = {
    "direct": "Direct Report",
    "news": "News",
    "social": "X",
}
DEFAULT_SOURCE = "X"


def _safe_text_value(value) -> str:
    """Return safe string value, treating NaN/None as empty."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        if math.isnan(value):
            return ""
    except TypeError:
        pass
    return str(value)


def genre_to_source(genre: str) -> str:
    """Map database genre to display source. Unknown/social genres map to X."""
    normalized = _safe_text_value(genre).strip().lower() or "direct"
    return GENRE_TO_SOURCE.get(normalized, DEFAULT_SOURCE)


def _log_api_error(label: str, error: Exception):
    """Log API errors with request context."""
    context = format_request_context()
    logger.error("%s failed%s: %s", label, context, error)


def _safe_label_value(value) -> int:
    """Return 0/1 label, treating NaN/None/inf/invalid values as 0."""
    if value is None:
        return 0
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _safe_float_prob(value) -> float:
    """Return probability as float; treat NaN/None/inf/non-numeric as 0.0 to avoid JSON/serialization failures."""
    if value is None:
        return 0.0
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    try:
        result = float(value)
        return 0.0 if math.isnan(result) or math.isinf(result) else result
    except (TypeError, ValueError):
        return 0.0


# Category relationships for improved probability simulation
CATEGORY_RELATIONSHIPS = {
    # Parent-child relationships (if parent is present, child confidence increases)
    "aid_related": ["medical_help", "medical_products", "water", "food", "shelter"],
    "infrastructure_related": ["buildings", "electricity", "transport", "hospitals"],
    "weather_related": ["floods", "storm", "cold"],
    # Co-occurrence patterns (if one is present, related ones get boost)
    "medical_help": ["hospitals", "medical_products"],
    "hospitals": ["medical_help", "medical_products"],
    "buildings": ["infrastructure_related", "electricity"],
    "electricity": ["infrastructure_related", "buildings"],
    "water": ["food", "shelter"],
    "food": ["water", "shelter"],
}

# Keywords that suggest higher confidence for categories
CATEGORY_KEYWORDS = {
    "medical_help": ["hospital", "doctor", "medical", "injured", "wound", "sick", "illness"],
    "medical_products": ["medicine", "supplies", "medication", "drugs", "bandage"],
    "water": ["water", "thirst", "drink", "hydrate"],
    "food": ["food", "hunger", "starving", "eat", "meal"],
    "shelter": ["shelter", "home", "house", "building", "roof"],
    "search_and_rescue": ["missing", "rescue", "search", "trapped", "stuck"],
    "infrastructure_related": ["road", "bridge", "building", "destroyed", "damage"],
    "buildings": ["building", "house", "structure", "destroyed", "collapsed"],
    "electricity": ["power", "electric", "light", "generator"],
    "weather_related": ["weather", "storm", "rain", "wind", "hurricane"],
    "storm": ["storm", "hurricane", "wind", "rain"],
    "floods": ["flood", "water", "drowned"],
    "fire": ["fire", "burning", "smoke"],
    "earthquake": ["earthquake", "shake", "tremor"],
}


def _improved_simulated_probabilities(row, category_columns: list, message: str = "") -> dict:
    """
    Build more realistic probabilities from binary labels using heuristics.
    
    Uses category relationships, keyword matching, and critical category
    weighting to generate more realistic probability distributions.
    """
    message_lower = (message or "").lower()
    probabilities = {}
    
    # Count how many categories are positive for this message
    positive_count = sum(1 for col in category_columns if _safe_label_value(row.get(col, 0)) == 1)
    
    for col in category_columns:
        label = _safe_label_value(row.get(col, 0))
        
        if label == 1:
            # Base probability for positive labels - higher for critical categories
            if col in CRITICAL_INTERNAL_CATEGORIES:
                base_prob = 0.80  # Critical categories get higher base
            else:
                base_prob = 0.70  # Non-critical positive labels
            
            # Boost if related categories are also present
            boost = 0.0
            if col in CATEGORY_RELATIONSHIPS:
                related_cats = CATEGORY_RELATIONSHIPS[col]
                related_count = sum(
                    1 for related_cat in related_cats
                    if _safe_label_value(row.get(related_cat, 0)) == 1
                )
                if related_count > 0:
                    boost += min(0.15, related_count * 0.05)  # Up to 15% boost
            
            # Boost if parent category is present (e.g., aid_related -> medical_help)
            for parent, children in CATEGORY_RELATIONSHIPS.items():
                if col in children and _safe_label_value(row.get(parent, 0)) == 1:
                    boost += 0.08
            
            # Boost if keywords match message content
            if col in CATEGORY_KEYWORDS:
                keywords = CATEGORY_KEYWORDS[col]
                matches = sum(1 for keyword in keywords if keyword in message_lower)
                if matches > 0:
                    boost += min(0.10, matches * 0.03)  # Up to 10% boost for keyword matches
            
            # Adjust based on how many categories are positive (more = slightly lower individual)
            if positive_count > 5:
                base_prob -= 0.05  # Slight reduction when many categories
            
            final_prob = base_prob + boost
            # Add small random variation (±5%)
            probabilities[col] = max(0.5, min(0.98, final_prob + random.uniform(-0.05, 0.05)))
        
        else:
            # For negative labels, use lower probabilities but with some variation
            # Messages with many positive categories might have slightly higher negatives
            if positive_count > 3:
                base_prob = random.uniform(0.15, 0.30)  # Slightly higher when many positives
            else:
                base_prob = random.uniform(0.05, 0.20)  # Lower baseline
            
            # If keywords strongly suggest this category but label is 0, keep it low
            if col in CATEGORY_KEYWORDS:
                keywords = CATEGORY_KEYWORDS[col]
                matches = sum(1 for keyword in keywords if keyword in message_lower)
                if matches > 2:  # Strong keyword match but label=0
                    base_prob = random.uniform(0.20, 0.35)  # Slightly higher but still below threshold
            
            probabilities[col] = base_prob
    
    return probabilities


def _simulated_probabilities(row, category_columns: list) -> dict:
    """
    Legacy wrapper for backward compatibility with tests.
    
    Calls _improved_simulated_probabilities without message context.
    """
    return _improved_simulated_probabilities(row, category_columns, "")


def _row_to_feed_item(row, category_columns: list) -> dict:
    """Convert a database row to a SignalItem dict for the feed."""
    raw_id = row.get("id", 0)
    msg = _safe_text_value(row.get("message")).strip()
    original = row.get("original")
    if hasattr(original, "strip"):
        original = (original or "").strip() or None
    else:
        original = None
    is_translated = bool(original and original != msg)
    content_preview = (msg[:120] + "...") if len(msg) > 120 else msg

    probabilities = _improved_simulated_probabilities(row, category_columns, msg)
    risk_level = calculate_severity(probabilities)

    # Only show categories that actually have label=1 in the training data
    # Filter to categories with actual positive labels before sorting
    labeled_cats = [
        (internal, conf)
        for internal, conf in probabilities.items()
        if _safe_label_value(row.get(internal, 0)) == 1
    ]
    sorted_cats = sorted(
        labeled_cats,
        key=lambda x: -x[1],
    )
    top_three = [to_display_name(internal) for internal, _ in sorted_cats[:3]]
    # Classifications should only include categories with actual label=1 and probability > 0.5
    classifications = [
        {"category": to_display_name(internal), "confidence": round(_safe_float_prob(conf), 2)}
        for internal, conf in sorted_cats
        if _safe_float_prob(conf) > 0.5
    ][:10]

    genre = row.get("genre")
    ts = generate_timestamp_for_id(raw_id)
    timestamp_iso = ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    return {
        "id": f"SIG-{raw_id}",
        "timestamp": timestamp_iso,
        "source": genre_to_source(genre),
        "content": content_preview,
        "originalContent": original if is_translated else None,
        "language": "en",
        "riskLevel": risk_level,
        "categories": top_three,
        "classifications": classifications,
        "isTranslated": is_translated,
    }


TREND_LABELS = [
    "6h ago",
    "5h ago",
    "4h ago",
    "3h ago",
    "2h ago",
    "1h ago",
    "Now",
]


def _build_metrics_response(df, category_columns: list) -> dict:
    """Build SYSTEM_METRICS from real category counts and simulated volume/trends."""
    n = len(df) if df is not None and not df.empty else 0
    cats = category_columns or []

    vol_today = n * 100 if n > 0 else 0
    if n > 0 and cats:
        filled = df[cats].fillna(0)
        flagged = (filled.sum(axis=1) > 0).sum()
        flagged_pct = round(min(10.0, 2.0 + (float(flagged) / n) * 5.0), 1)
        sums = filled.sum()
        top = sums.sort_values(ascending=False).head(7)
        top_categories = [
            {"name": to_display_name(internal), "count": _safe_label_value(count)}
            for internal, count in top.items()
        ]
    else:
        flagged_pct = 0.0
        top_categories = []

    if n > 0:
        trend_counts = [45, 120, 80, 210, 150, 95, 60]
    else:
        trend_counts = [0, 0, 0, 0, 0, 0, 0]
    trend_data = [{"time": label, "count": c} for label, c in zip(TREND_LABELS, trend_counts)]

    return {
        "volToday": vol_today,
        "flaggedRate": flagged_pct,
        "topCategories": top_categories,
        "trendData": trend_data,
    }


def _build_stub_classification() -> dict:
    """Create classification results stub for contract validation."""
    return {
        "categories": [
            {"name": "Water", "confidence": 0.92, "volume": 892},
            {"name": "Search & Rescue", "confidence": 0.88, "volume": 421},
        ],
        "severity": "HIGH",
        "maxConfidence": 0.92,
        "avgConfidence": 0.90,
    }


def _build_simplified_classification(
    prediction_result: dict,
    category_volumes: dict,
    thresholds_map: dict,
) -> dict:
    """Build simplified classification response with severity and volume context."""
    probabilities = prediction_result.get("probabilities") or {}
    if not probabilities:
        return {
            "categories": [],
            "severity": "LOW",
            "maxConfidence": 0.0,
            "avgConfidence": 0.0,
        }
    threshold_default = 0.5
    above_threshold = [
        (internal, prob)
        for internal, prob in probabilities.items()
        if _safe_float_prob(prob)
        >= _safe_float_prob(thresholds_map.get(internal, threshold_default))
    ]
    above_threshold.sort(key=lambda x: -_safe_float_prob(x[1]))
    categories = [
        {
            "name": _safe_category_display(internal),
            "confidence": round(_safe_float_prob(prob), 2),
            "volume": _safe_label_value(category_volumes.get(internal, 0)),
        }
        for internal, prob in above_threshold[:10]
    ]
    severity = calculate_severity(probabilities)
    returned_probs = [_safe_float_prob(prob) for _, prob in above_threshold[:10]]
    max_conf = round(max(returned_probs), 2) if returned_probs else 0.0
    avg_conf = (
        round(sum(returned_probs) / len(returned_probs), 2) if returned_probs else 0.0
    )
    return {
        "categories": categories,
        "severity": severity,
        "maxConfidence": max_conf,
        "avgConfidence": avg_conf,
    }


def _get_feed_filter_categories() -> list:
    """Parse categories[] query param (internal names) for feed filter."""
    names = request.args.getlist("categories[]") or request.args.getlist("categories")
    return [n.strip() for n in names if n and isinstance(n, str)]


@api_bp.route("/feed", methods=["GET"])
def feed():
    """Return paginated feed items from the database (binary labels + simulated confidences)."""
    try:
        data_service = getattr(current_app, "data_service", None)
        if data_service is None:
            raise DataServiceError("Data service not configured.")
        df = data_service.get_data()
        category_columns = data_service.get_category_columns()
        if not category_columns:
            category_columns = []

        # Filter to only show messages that are disaster-related (related=1)
        # related can be 0 (not related), 1 (related), or 2 (unclassifiable)
        if "related" in df.columns:
            df = df.loc[df["related"] == 1].copy()

        # Filter out meta-categories from category_columns before processing
        # "related" is a meta-category indicating disaster-relevance, not a specific category
        displayable_category_columns = [col for col in category_columns if col != "related"]

        limit_raw = request.args.get("limit", 25, type=int)
        offset_raw = request.args.get("offset", 0, type=int)
        limit = min(max(1, limit_raw if limit_raw is not None else 25), 100)
        offset = max(0, offset_raw if offset_raw is not None else 0)
        filter_cats = _get_feed_filter_categories()

        if filter_cats:
            valid_cats = [c for c in filter_cats if c in displayable_category_columns]
            if valid_cats:
                mask = df[valid_cats].sum(axis=1) > 0
                df = df.loc[mask]
        total = len(df)
        if total == 0:
            page = 1
            total_pages = 0
            effective_offset = 0
        else:
            page = (offset // limit) + 1
            total_pages = (total + limit - 1) // limit
            if offset >= total:
                page = total_pages
                effective_offset = (total_pages - 1) * limit
            else:
                effective_offset = offset
        slice_df = df.iloc[effective_offset : effective_offset + limit]

        items = []
        for _, row in slice_df.iterrows():
            item = _row_to_feed_item(row.to_dict(), displayable_category_columns)
            items.append(item)

        payload = {
            "items": items,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "totalPages": total_pages,
            },
        }
        return jsonify(payload)
    except Exception as error:
        _log_api_error("GET /api/feed", error)
        return jsonify({"error": "Feed unavailable right now."}), 500


@api_bp.route("/metrics", methods=["GET"])
def metrics():
    """Return metrics with real category counts and simulated volume/trends."""
    try:
        data_service = getattr(current_app, "data_service", None)
        if data_service is None:
            raise DataServiceError("Data service not configured.")
        df = data_service.get_data()
        category_columns = data_service.get_category_columns()
        if not category_columns:
            category_columns = []
        payload = _build_metrics_response(df, category_columns)
        return jsonify(payload)
    except Exception as error:
        _log_api_error("GET /api/metrics", error)
        return jsonify({"error": "Metrics unavailable right now."}), 500


@api_bp.route("/categories", methods=["GET"])
def categories_metadata():
    """Return category metadata for dashboard filters."""
    try:
        data_service = getattr(current_app, "data_service", None)
        if data_service is None:
            raise DataServiceError("Data service not configured.")
        df = data_service.get_data()
        category_columns = data_service.get_category_columns() or []
        counts = (
            df[category_columns].fillna(0).sum().to_dict()
            if category_columns and not df.empty
            else {col: 0 for col in category_columns}
        )
        categories = [
            {
                "internal": internal,
                "display": to_display_name(internal),
                "count": _safe_label_value(counts.get(internal, 0)),
            }
            for internal in sorted(category_columns)
        ]
        return jsonify({"categories": categories, "groups": CATEGORY_GROUPS})
    except Exception as error:
        _log_api_error("GET /api/categories", error)
        return jsonify({"error": "Categories unavailable right now."}), 500


@api_bp.route("/dashboard")
@api_bp.route("/dashboard/<path:path>")
def dashboard(path=None):
    """Serve Storm Signal dashboard SPA (React app)."""
    static_folder = current_app.static_folder
    return send_from_directory(
        static_folder, "dashboard/index.html", mimetype="text/html"
    )


@api_bp.route("/classify", methods=["POST"])
def classify():
    """Return simplified classification with severity and category volume context."""
    try:
        body = request.get_json(silent=True) or {}
        message = (body.get("message") or "").strip()
        if not message:
            return jsonify({"error": "Message is required."}), 400

        model_service = getattr(current_app, "model_service", None)
        if model_service is None:
            return jsonify({"error": "Classification unavailable right now."}), 503

        prediction_result = process_prediction_result(model_service, message)
        if not prediction_result.get("is_valid", True):
            return (
                jsonify({"error": prediction_result.get("error_message", "Invalid message.")}),
                400,
            )

        category_volumes = {}
        data_service = getattr(current_app, "data_service", None)
        if data_service is not None:
            try:
                df = data_service.get_data()
                category_columns = data_service.get_category_columns() or []
                if category_columns and df is not None and not df.empty:
                    sums = df[category_columns].fillna(0).sum()
                    category_volumes = sums.to_dict()
            except (DataServiceError, Exception) as data_error:
                _log_api_error("POST /api/classify (data_service)", data_error)
                # Volumes are supplementary; classification continues with empty volumes

        thresholds_map = model_service.get_thresholds_map()
        payload = _build_simplified_classification(
            prediction_result, category_volumes, thresholds_map
        )
        return jsonify(payload)
    except (ValueError, ModelServiceError) as error:
        _log_api_error("POST /api/classify", error)
        return jsonify({"error": "Classification unavailable right now."}), 503
    except Exception as error:
        _log_api_error("POST /api/classify", error)
        return jsonify({"error": "Classification unavailable right now."}), 500
