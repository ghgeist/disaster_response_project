"""
API contract stubs for the Storm Signal dashboard.
"""
import logging
import re
from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify

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
    re.sub(r"\s+", "_", re.sub(r"[^a-zA-Z0-9\s]", " ", name.replace("&", "and")))
    .strip()
    .lower()
    for name in CATEGORY_GROUPS["Critical Needs"]
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


def _now_iso() -> str:
    """Return a UTC timestamp string for stubs."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _log_api_error(label: str, error: Exception):
    """Log API errors with request context."""
    context = format_request_context()
    logger.error("%s failed%s: %s", label, context, error)


def _build_stub_feed_item() -> dict:
    """Create a single SignalItem stub for contract validation."""
    display_categories = [
        to_display_name("water"),
        to_display_name("search_and_rescue"),
        to_display_name("floods"),
    ]
    return {
        "id": "SIG-1001",
        "timestamp": _now_iso(),
        "source": "Twitter",
        "content": "Urgent: Water rising rapidly near the east bridge.",
        "originalContent": None,
        "language": "en",
        "riskLevel": "HIGH",
        "categories": display_categories,
        "classifications": [
            {"category": display_categories[0], "confidence": 0.92},
            {"category": display_categories[1], "confidence": 0.88},
        ],
        "isTranslated": False,
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


@api_bp.route("/feed", methods=["GET"])
def feed_stub():
    """Return a stubbed feed response for contract validation."""
    try:
        items = [_build_stub_feed_item()]
        payload = {
            "items": items,
            "pagination": {
                "page": 1,
                "limit": 25,
                "total": 150,
                "totalPages": 6,
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
