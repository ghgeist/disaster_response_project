"""
API contract stubs for the Storm Signal dashboard.
"""
import hashlib
import logging
import math
import random
from datetime import datetime, timedelta, timezone

from flask import Blueprint, current_app, jsonify, request

from app.extensions import csrf
from app.services.errors import DataServiceError
from app.utils.formatting import format_request_context

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


def calculate_severity(probabilities: dict) -> str:
    """Determine severity based on critical category probabilities."""
    critical_count = sum(
        1
        for category, probability in probabilities.items()
        if category in CRITICAL_INTERNAL_CATEGORIES and probability > 0.5
    )
    critical_probabilities = [
        probability
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


def genre_to_source(genre: str) -> str:
    """Map database genre to display source. Unknown/social genres map to X."""
    normalized = (genre or "").strip().lower() or "direct"
    return GENRE_TO_SOURCE.get(normalized, DEFAULT_SOURCE)


def _log_api_error(label: str, error: Exception):
    """Log API errors with request context."""
    context = format_request_context()
    logger.error("%s failed%s: %s", label, context, error)


def _simulated_probabilities(row, category_columns: list) -> dict:
    """Build probabilities from binary labels with clear separation."""
    return {
        col: 0.1 + (_safe_label_value(row.get(col, 0)) * 0.6) + random.uniform(0, 0.1)
        for col in category_columns
    }


def _safe_label_value(value) -> int:
    """Return 0/1 label, treating NaN/None/invalid values as 0."""
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _row_to_feed_item(row, category_columns: list) -> dict:
    """Convert a database row to a SignalItem dict for the feed."""
    raw_id = row.get("id", 0)
    msg = (row.get("message") or "").strip()
    original = row.get("original")
    if hasattr(original, "strip"):
        original = (original or "").strip() or None
    else:
        original = None
    is_translated = bool(original and original != msg)
    content_preview = (msg[:120] + "...") if len(msg) > 120 else msg

    probabilities = _simulated_probabilities(row, category_columns)
    risk_level = calculate_severity(probabilities)
    sorted_cats = sorted(
        probabilities.items(),
        key=lambda x: -x[1],
    )
    top_three = [to_display_name(internal) for internal, _ in sorted_cats[:3]]
    classifications = [
        {"category": to_display_name(internal), "confidence": round(conf, 2)}
        for internal, conf in sorted_cats
        if conf > 0.5
    ][:10]

    genre = row.get("genre") or "direct"
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


def _build_stub_metrics() -> dict:
    """Create a SYSTEM_METRICS stub for contract validation."""
    return {
        "volToday": 14502,
        "flaggedRate": 4.2,
        "topCategories": [
            {"name": "Medical Help", "count": 1247},
            {"name": "Water", "count": 892},
            {"name": "Food", "count": 756},
        ],
        "trendData": [
            {"time": "6h ago", "count": 45},
            {"time": "5h ago", "count": 120},
            {"time": "4h ago", "count": 80},
            {"time": "3h ago", "count": 210},
            {"time": "2h ago", "count": 150},
            {"time": "1h ago", "count": 95},
            {"time": "Now", "count": 60},
        ],
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

        limit = min(max(1, request.args.get("limit", 25, type=int)), 100)
        offset = max(0, request.args.get("offset", 0, type=int))
        filter_cats = _get_feed_filter_categories()

        if filter_cats:
            valid_cats = [c for c in filter_cats if c in category_columns]
            if valid_cats:
                mask = df[valid_cats].sum(axis=1) > 0
                df = df.loc[mask]
        total = len(df)
        if total == 0:
            page = 1
            total_pages = 0
        else:
            page = (offset // limit) + 1
            total_pages = (total + limit - 1) // limit
        slice_df = df.iloc[offset : offset + limit]

        items = []
        for _, row in slice_df.iterrows():
            item = _row_to_feed_item(row.to_dict(), category_columns)
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
def metrics_stub():
    """Return a stubbed metrics response for contract validation."""
    try:
        return jsonify(_build_stub_metrics())
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
        category_columns = data_service.get_category_columns()
        counts = (
            df[category_columns].sum().to_dict()
            if category_columns and not df.empty
            else {col: 0 for col in category_columns}
        )
        categories = [
            {
                "internal": internal,
                "display": to_display_name(internal),
                "count": int(counts.get(internal, 0)),
            }
            for internal in sorted(category_columns)
        ]
        return jsonify({"categories": categories, "groups": CATEGORY_GROUPS})
    except Exception as error:
        _log_api_error("GET /api/categories", error)
        return jsonify({"error": "Categories unavailable right now."}), 500


@api_bp.route("/classify", methods=["POST"])
def classify_stub():
    """Return stubbed classification results for contract validation."""
    try:
        return jsonify(_build_stub_classification())
    except Exception as error:
        _log_api_error("POST /api/classify", error)
        return jsonify({"error": "Classification unavailable right now."}), 500
