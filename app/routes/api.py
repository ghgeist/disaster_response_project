"""
Storm Signal dashboard API: feed, metrics, categories, and classification.
"""
import hashlib
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Blueprint, current_app, jsonify, redirect, request

from app.extensions import csrf
from app.services.demo_feed import (
    DEFAULT_DEMO_FEED_PATH,
    assert_demo_feed_matches_production,
    load_demo_feed,
)
from app.services.errors import DataServiceError, DemoFeedError
from app.services.model_service import ModelServiceError
from app.services.production_artifacts import (
    ProductionArtifactError,
    resolve_production_artifacts,
    stem_bound_thresholds_path,
)
from app.utils.feed_display import (  # noqa: F401
    CATEGORY_DISPLAY_NAMES,
    CRITICAL_INTERNAL_CATEGORIES,
    DEFAULT_SOURCE,
    GENRE_TO_SOURCE,
    _safe_float_prob,
    _safe_text_value,
    calculate_severity,
    genre_to_source,
    to_display_name,
)
from app.utils.formatting import format_request_context
from app.utils.hierarchy_helpers import run_hierarchy_correction
from app.utils.prediction_helpers import process_prediction_result
from app.utils.validation import validate_message_text
from disasterproject.utils.config import TAXONOMY

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")
csrf.exempt(api_bp)

# Re-export display helpers/constants for existing tests:
# ``from app.routes.api import calculate_severity, to_display_name, ...``
__all_display_reexports__ = (
    "CATEGORY_DISPLAY_NAMES",
    "CRITICAL_INTERNAL_CATEGORIES",
    "DEFAULT_SOURCE",
    "GENRE_TO_SOURCE",
    "_safe_float_prob",
    "_safe_text_value",
    "calculate_severity",
    "genre_to_source",
    "to_display_name",
)

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
        "Missing People",
        "Refugees",
        "Death",
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

HIERARCHY_UNGROUPED_KEY = "ungrouped"
HIERARCHY_UNGROUPED_LABEL = "Ungrouped"
TAXONOMY_CHILD_TO_PARENT = {
    child: parent for parent, children in TAXONOMY.items() for child in children
}


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


def generate_timestamp_for_id(raw_id) -> datetime:
    """Return a stable timestamp for a record based on its id (last 6 hours, deterministic)."""
    key = str(raw_id).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    fraction = int(digest[:12], 16) / (16**12)
    hours_ago = fraction * 6
    return datetime.now(timezone.utc) - timedelta(hours=hours_ago)


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


def _safe_optional_prob(value) -> float | None:
    """Return a finite probability in [0, 1], else None.

    Unlike ``_safe_float_prob``, missing/malformed/NaN/inf/out-of-range values
    become None so callers do not treat corruption as a legitimate 0.0.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    if result < 0.0 or result > 1.0:
        return None
    return result


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
    """Build simplified classification response with severity and volume context.

    When ``labels`` are present (hierarchy-corrected binaries), membership uses those
    decisions so child→parent activations are not dropped when parent probability is
    still below the parent threshold. Probabilities drive confidence and severity.
    """
    probabilities = prediction_result.get("probabilities") or {}
    if not probabilities:
        return {
            "categories": [],
            "severity": "LOW",
            "maxConfidence": 0.0,
            "avgConfidence": 0.0,
        }
    threshold_default = 0.5
    labels = prediction_result.get("labels") or {}
    # Exclude meta-category "related" (disaster-relevance) from detection results
    if labels:
        selected = [
            (internal, probabilities.get(internal, 0.0))
            for internal, label in labels.items()
            if internal != "related" and _safe_label_value(label) == 1
        ]
    else:
        selected = [
            (internal, prob)
            for internal, prob in probabilities.items()
            if internal != "related"
            and _safe_float_prob(prob)
            >= _safe_float_prob(thresholds_map.get(internal, threshold_default))
        ]
    selected.sort(key=lambda x: -_safe_float_prob(x[1]))
    categories = []
    for internal, prob in selected[:10]:
        threshold = _safe_float_prob(thresholds_map.get(internal, threshold_default))
        # Positive decisions (including hierarchy-forced parents) meet the decision
        # contract even when adjusted probability is still below the parent threshold.
        if labels:
            meets_threshold = _safe_label_value(labels.get(internal, 0)) == 1
        else:
            meets_threshold = _safe_float_prob(prob) >= threshold
        categories.append(
            {
                "name": _safe_category_display(internal),
                "confidence": round(_safe_float_prob(prob), 2),
                "volume": _safe_label_value(category_volumes.get(internal, 0)),
                "threshold": round(threshold, 3),
                "meetsThreshold": meets_threshold,
            }
        )
    severity = calculate_severity(probabilities)
    returned_probs = [_safe_float_prob(prob) for _, prob in selected[:10]]
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


def _prepare_displayable_data(df, category_columns: list):
    """Return filtered df and displayable category columns for dashboard endpoints."""
    if not category_columns:
        category_columns = []
    # Filter to only show messages that are disaster-related (related=1)
    # related can be 0 (not related), 1 (related), or 2 (unclassifiable)
    if df is not None and "related" in df.columns:
        df = df.loc[df["related"] == 1].copy()
    # Filter out meta-categories before processing
    # "related" is a meta-category indicating disaster-relevance, not a specific category
    displayable_category_columns = [col for col in category_columns if col != "related"]
    return df, displayable_category_columns


def _item_matches_category_filter(item: dict, filter_cats: list) -> bool:
    """True when any requested internal name is a hierarchy-corrected positive."""
    labels = (item.get("fixed") or {}).get("labels") or {}
    return any(_safe_label_value(labels.get(cat, 0)) == 1 for cat in filter_cats)


def _shape_public_feed_item(cached_item: dict) -> dict:
    """Project a cached demo-feed item to the public feed contract."""
    message_id = cached_item.get("message_id", cached_item.get("id", 0))
    ts = generate_timestamp_for_id(message_id)
    timestamp_iso = ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "id": cached_item.get("id"),
        "timestamp": timestamp_iso,
        "source": cached_item.get("source"),
        "content": cached_item.get("content"),
        "originalContent": cached_item.get("originalContent"),
        "language": cached_item.get("language", "en"),
        "riskLevel": cached_item.get("riskLevel"),
        "categories": list(cached_item.get("categories") or []),
        "classifications": list(cached_item.get("classifications") or []),
        "isTranslated": bool(cached_item.get("isTranslated")),
    }


def _paginate_feed_items(items: list, limit: int, offset: int) -> tuple[list, dict]:
    """Apply existing limit/offset clamps and return the page plus pagination meta."""
    total = len(items)
    if total == 0:
        return [], {"page": 1, "limit": limit, "total": 0, "totalPages": 0}

    page = (offset // limit) + 1
    total_pages = (total + limit - 1) // limit
    if offset >= total:
        page = total_pages
        effective_offset = (total_pages - 1) * limit
    else:
        effective_offset = offset
    page_items = items[effective_offset : effective_offset + limit]
    return page_items, {
        "page": page,
        "limit": limit,
        "total": total,
        "totalPages": total_pages,
    }


@api_bp.route("/feed", methods=["GET"])
def feed():
    """Return paginated feed items from the production-classified demo-feed cache."""
    try:
        model_service = getattr(current_app, "model_service", None)
        if model_service is None:
            raise ModelServiceError("Model service not configured.")

        feed_path = current_app.config.get("DEMO_FEED_PATH", DEFAULT_DEMO_FEED_PATH)
        cached = load_demo_feed(feed_path)
        artifacts = model_service.get_production_artifacts()
        assert_demo_feed_matches_production(cached, artifacts)

        limit_raw = request.args.get("limit", 25, type=int)
        offset_raw = request.args.get("offset", 0, type=int)
        limit = min(max(1, limit_raw if limit_raw is not None else 25), 100)
        offset = max(0, offset_raw if offset_raw is not None else 0)
        filter_cats = _get_feed_filter_categories()

        items = list(cached.get("items") or [])
        if filter_cats:
            items = [item for item in items if _item_matches_category_filter(item, filter_cats)]

        page_items, pagination = _paginate_feed_items(items, limit, offset)
        payload = {
            "items": [_shape_public_feed_item(item) for item in page_items],
            "pagination": pagination,
        }
        return jsonify(payload)
    except (DemoFeedError, ModelServiceError, FileNotFoundError, OSError, ValueError) as error:
        _log_api_error("GET /api/feed", error)
        return jsonify({"error": "Feed unavailable right now."}), 503
    except Exception as error:
        _log_api_error("GET /api/feed", error)
        return jsonify({"error": "Feed unavailable right now."}), 503


@api_bp.route("/metrics", methods=["GET"])
def metrics():
    """Return metrics with real category counts and simulated volume/trends."""
    try:
        data_service = getattr(current_app, "data_service", None)
        if data_service is None:
            raise DataServiceError("Data service not configured.")
        df = data_service.get_data()
        category_columns = data_service.get_category_columns()
        df, displayable_category_columns = _prepare_displayable_data(df, category_columns)

        payload = _build_metrics_response(df, displayable_category_columns)
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
        df, displayable_category_columns = _prepare_displayable_data(df, category_columns)

        counts = (
            df[displayable_category_columns].fillna(0).sum().to_dict()
            if displayable_category_columns and not df.empty
            else {col: 0 for col in displayable_category_columns}
        )
        categories = [
            {
                "internal": internal,
                "display": to_display_name(internal),
                "count": _safe_label_value(counts.get(internal, 0)),
            }
            for internal in sorted(displayable_category_columns)
        ]
        return jsonify({"categories": categories, "groups": CATEGORY_GROUPS})
    except Exception as error:
        _log_api_error("GET /api/categories", error)
        return jsonify({"error": "Categories unavailable right now."}), 500


def _get_model_dir() -> Path:
    """Return path to project model/ directory."""
    return Path(current_app.root_path).parent / "model"


def _resolve_active_production_model_path(model_dir: Path) -> Path | None:
    """
    Resolve the active production model path (same contract as inference).

    Prefer ``current_app.config['MODEL_PATH']`` when it points at an existing
    ``disaster_*_prod_*.pkl`` under ``model_dir``. Otherwise fall back to
    newest-by-mtime discovery for that pattern.
    """
    if not model_dir.is_dir():
        return None

    configured = current_app.config.get("MODEL_PATH")
    if configured is not None:
        configured_path = Path(configured)
        if (
            configured_path.is_file()
            and configured_path.parent.resolve() == model_dir.resolve()
            and configured_path.name.startswith("disaster_")
            and "_prod_" in configured_path.name
            and configured_path.suffix == ".pkl"
        ):
            return configured_path

    model_files = list(model_dir.glob("disaster_*_prod_*.pkl"))
    if not model_files:
        return None
    model_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return model_files[0]


def _find_production_thresholds_file(
    model_dir: Path, model_stem: str | None = None
) -> Path | None:
    """
    Find production thresholds for the active model stem.

    Binding is by filename stem (``{model_stem}_thresholds.json``), matching
    the shared production-artifact resolver used by inference — never
    newest-by-mtime across orphans and never legacy ``thresholds.json``.

    ``metadata.model`` inside the JSON is training-source provenance and may
    still name the experimental candidate; it is not used for discovery.
    """
    if not model_dir.is_dir():
        return None

    stem = model_stem
    if not stem or stem == "unknown":
        active_model = _resolve_active_production_model_path(model_dir)
        if active_model is None:
            return None
        stem = active_model.stem

    stem_thresholds = stem_bound_thresholds_path(model_dir / f"{stem}.pkl")
    if stem_thresholds.is_file() and not stem_thresholds.name.startswith("optimized_"):
        return stem_thresholds
    return None


def _load_thresholds_json(thresholds_path: Path) -> dict:
    """Load thresholds JSON already validated by the production-artifact resolver."""
    try:
        with open(thresholds_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Thresholds file read failed %s: %s", thresholds_path, error)
        return {}
    if not isinstance(payload, dict):
        logger.warning("Thresholds file is not a JSON object: %s", thresholds_path)
        return {}
    return payload


def _category_stats_list_from_thresholds(thresholds_payload: dict) -> list:
    """Return category_stats from the hashed thresholds artifact (empty if absent)."""
    raw_stats = thresholds_payload.get("category_stats")
    if not isinstance(raw_stats, list):
        return []
    return [stat for stat in raw_stats if isinstance(stat, dict)]


def _support_weighted_precision_recall(
    category_stats_list: list,
) -> tuple[float | None, float | None]:
    """Support-weighted means of positive-class precision/recall; null when unmeasured."""
    total_support = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    for stat in category_stats_list:
        support_val = stat.get("support")
        try:
            support = float(support_val) if support_val is not None else 0.0
        except (TypeError, ValueError):
            support = 0.0
        if math.isnan(support) or math.isinf(support) or support < 0:
            support = 0.0
        total_support += support
        precision = _safe_float_prob(stat.get("precision"))
        if "actual_recall" in stat:
            recall = _safe_float_prob(stat.get("actual_recall"))
        else:
            recall = _safe_float_prob(stat.get("recall"))
        weighted_precision += precision * support
        weighted_recall += recall * support

    if total_support <= 0:
        return None, None
    precision_overall = weighted_precision / total_support
    recall_overall = weighted_recall / total_support
    if math.isnan(precision_overall) or math.isinf(precision_overall):
        precision_overall = None
    if math.isnan(recall_overall) or math.isinf(recall_overall):
        recall_overall = None
    return precision_overall, recall_overall


DASHBOARD_PROVENANCE_UNAVAILABLE = (
    "Production model provenance unavailable"
)


def _unavailable_dashboard_payload(
    *,
    generated_at: str,
    stem: str = "unknown",
    provenance_code: str = "provenance_unavailable",
) -> dict:
    """Stable unavailable payload when production artifacts cannot be resolved."""
    model_id = stem.upper().replace("-", "_") if stem != "unknown" else stem
    return {
        "model": {
            "id": model_id,
            "version": "unknown",
            "lastUpdated": None,
            "status": "unavailable",
            "generatedAt": generated_at,
            "algorithm": "unknown",
            "algorithmName": "Unknown",
            "provenanceError": DASHBOARD_PROVENANCE_UNAVAILABLE,
            "provenanceCode": provenance_code,
        },
        "metrics": {
            "f1": None,
            "precision": None,
            "recall": None,
            "evalCriticalRecall": None,
        },
        "categories": [],
        "criticalThresholds": [],
        "registry": [],
    }


def _build_model_info_dashboard_payload() -> dict:
    """
    Build single payload for Model Information dashboard.

    Uses the shared production-artifact resolver whenever an active production
    pickle exists. Reports unavailable when no pickle is present or provenance
    fails — never surfaces orphan MODEL_INFO metadata alone.
    """
    model_dir = _get_model_dir()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    active_model = _resolve_active_production_model_path(model_dir)
    if active_model is None:
        return _unavailable_dashboard_payload(
            generated_at=generated_at,
            provenance_code="active_model_missing",
        )

    try:
        production_artifacts = resolve_production_artifacts(active_model)
        model_info_data = dict(production_artifacts.model_info)
    except ProductionArtifactError as error:
        logger.warning(
            "Production artifact provenance failed for dashboard: %s", error
        )
        return _unavailable_dashboard_payload(
            generated_at=generated_at,
            stem=active_model.stem,
            provenance_code="provenance_failed",
        )

    version = model_info_data.get("version", "unknown")
    if not isinstance(version, str):
        version = "unknown"
    last_updated = model_info_data.get("promotion_timestamp")
    if last_updated is None or (
        isinstance(last_updated, float) and (math.isnan(last_updated) or math.isinf(last_updated))
    ):
        last_updated = None
    else:
        last_updated = str(last_updated)
    status = model_info_data.get("status", "unknown")
    if not isinstance(status, str):
        status = "unknown"

    performance_block = model_info_data.get("performance") or {}
    validation_block = model_info_data.get("validation_results") or {}
    # Explicit OP vocabulary only — never fall back to naked f1_weighted.
    # Missing/invalid → null (not 0.0) so UI can render "—".
    optimized_f1_weighted = performance_block.get("optimized_f1_weighted")
    if optimized_f1_weighted is None:
        optimized_f1_weighted = validation_block.get("optimized_f1_weighted")
    f1_metric = _safe_optional_prob(optimized_f1_weighted)

    eval_critical_raw = performance_block.get("eval_critical_recall")
    if eval_critical_raw is None:
        eval_critical_raw = validation_block.get("eval_critical_recall")
    eval_critical_recall = _safe_optional_prob(eval_critical_raw)

    stem = active_model.stem
    inference_thresholds = production_artifacts.thresholds
    thresholds_payload = _load_thresholds_json(production_artifacts.paths.thresholds_path)
    category_stats_list = _category_stats_list_from_thresholds(thresholds_payload)

    critical_thresholds_list: list = []
    categories_payload: list = []
    precision_overall: float | None = None
    recall_overall: float | None = None

    if category_stats_list:
        precision_overall, recall_overall = _support_weighted_precision_recall(
            category_stats_list
        )
        for stat in category_stats_list:
            key = stat.get("category")
            if key is None:
                continue
            key_str = str(key)
            label = _safe_category_display(key_str)
            support_val = stat.get("support")
            try:
                support = float(support_val) if support_val is not None else 0.0
            except (TypeError, ValueError):
                support = 0.0
            if math.isnan(support) or math.isinf(support) or support < 0:
                support = 0.0
            precision = _safe_float_prob(stat.get("precision"))
            if "actual_recall" in stat:
                recall = _safe_float_prob(stat.get("actual_recall"))
            else:
                recall = _safe_float_prob(stat.get("recall"))
            parent_key = key_str if key_str in TAXONOMY else TAXONOMY_CHILD_TO_PARENT.get(key_str)
            if parent_key is None:
                parent_key = HIERARCHY_UNGROUPED_KEY
                parent_label = HIERARCHY_UNGROUPED_LABEL
            else:
                parent_label = _safe_category_display(parent_key)
            categories_payload.append({
                "key": key_str,
                "label": label,
                "f1": _safe_float_prob(stat.get("f1")),
                "precision": precision,
                "recall": recall,
                "support": int(support),
                "hierarchyParentKey": parent_key,
                "hierarchyParentLabel": parent_label,
            })
            if stat.get("type") == "critical":
                if key_str not in inference_thresholds:
                    logger.warning(
                        "Critical category %s missing from inference threshold map; "
                        "skipping criticalThresholds entry",
                        key_str,
                    )
                    continue
                critical_thresholds_list.append({
                    "key": key_str,
                    "label": label,
                    "threshold": _safe_float_prob(inference_thresholds[key_str]),
                })

    registry_allowlist = {".json", ".csv", ".md", ".pkl"}
    registry_list = []
    if model_dir.is_dir():
        for f in model_dir.iterdir():
            if not f.is_file():
                continue
            suf = f.suffix.lower()
            if suf not in registry_allowlist:
                continue
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            registry_list.append({
                "name": f.name,
                "size": size,
                "type": suf.lstrip("."),
            })
    registry_list.sort(key=lambda x: x["name"])

    model_id_upper = stem.upper().replace("-", "_") if stem != "unknown" else stem

    # Extract algorithm information from MODEL_INFO.json
    algorithm_code = model_info_data.get("algorithm", "unknown")
    algorithm_name = model_info_data.get("algorithm_name", "Unknown")

    return {
        "model": {
            "id": model_id_upper,
            "version": version,
            "lastUpdated": last_updated,
            "status": status,
            "generatedAt": generated_at,
            "algorithm": algorithm_code,
            "algorithmName": algorithm_name,
        },
        "metrics": {
            "f1": round(f1_metric, 4) if f1_metric is not None else None,
            "precision": (
                round(precision_overall, 4) if precision_overall is not None else None
            ),
            "recall": (
                round(recall_overall, 4) if recall_overall is not None else None
            ),
            "evalCriticalRecall": (
                round(eval_critical_recall, 4)
                if eval_critical_recall is not None
                else None
            ),
        },
        "categories": categories_payload,
        "criticalThresholds": critical_thresholds_list,
        "registry": registry_list,
    }


def _unavailable_model_info_payload(
    *,
    provenance_code: str = "provenance_unavailable",
) -> dict:
    """Stable unavailable payload for GET /api/model-info."""
    return {
        "version": "unknown",
        "f1_score": None,
        "status": "unavailable",
        "hierarchy_violations": 0.0,
        "provenanceError": DASHBOARD_PROVENANCE_UNAVAILABLE,
        "provenanceCode": provenance_code,
    }


@api_bp.route("/model-info", methods=["GET"])
def model_info():
    """Return production model metadata from a provenance-valid active bundle."""
    try:
        model_dir = _get_model_dir()
        active_model = _resolve_active_production_model_path(model_dir)
        if active_model is None:
            return jsonify(
                _unavailable_model_info_payload(provenance_code="active_model_missing")
            )

        try:
            production_artifacts = resolve_production_artifacts(active_model)
            model_info_data = dict(production_artifacts.model_info)
        except ProductionArtifactError as error:
            logger.warning(
                "Production artifact provenance failed for /api/model-info: %s", error
            )
            return jsonify(
                _unavailable_model_info_payload(provenance_code="provenance_failed")
            )

        version = model_info_data.get("version", "unknown")
        if not isinstance(version, str):
            version = "unknown"

        performance_block = model_info_data.get("performance") or {}
        validation_block = model_info_data.get("validation_results") or {}
        # Explicit OP vocabulary only — never fall back to naked f1_weighted.
        optimized_f1_weighted = performance_block.get("optimized_f1_weighted")
        if optimized_f1_weighted is None:
            optimized_f1_weighted = validation_block.get("optimized_f1_weighted")
        try:
            f1_score = (
                float(optimized_f1_weighted)
                if optimized_f1_weighted is not None
                else None
            )
        except (TypeError, ValueError):
            f1_score = None
        if f1_score is not None and (math.isnan(f1_score) or math.isinf(f1_score)):
            f1_score = None

        status = model_info_data.get("status", "unknown")
        if not isinstance(status, str):
            status = "unknown"

        return jsonify({
            "version": version,
            "f1_score": f1_score,
            "status": status,
            "hierarchy_violations": 0.0,
        })
    except Exception as error:
        _log_api_error("GET /api/model-info", error)
        return jsonify({"error": "Model info unavailable right now."}), 500


@api_bp.route("/model-info/dashboard", methods=["GET"])
def model_info_dashboard():
    """Return single payload for Model Information dashboard (model, metrics, categories, criticalThresholds, registry)."""
    try:
        payload = _build_model_info_dashboard_payload()
        return jsonify(payload)
    except Exception as error:
        _log_api_error("GET /api/model-info/dashboard", error)
        return jsonify({"error": "Model info dashboard unavailable right now."}), 500


@api_bp.route("/model-info-dashboard", defaults={"path": ""})
@api_bp.route("/model-info-dashboard/<path:path>")
def model_info_dashboard_spa_redirect(path: str):
    """Redirect legacy SPA routes to the new production model dashboard."""
    target = f"/production-model/{path}" if path else "/production-model"
    return redirect(target)


@api_bp.route("/dashboard", defaults={"path": ""})
@api_bp.route("/dashboard/<path:path>")
def dashboard_redirect(path: str):
    """Redirect legacy dashboard routes to the new public dashboard."""
    target = f"/dashboard/{path}" if path else "/dashboard"
    return redirect(target)


@api_bp.route("/about", defaults={"path": ""})
@api_bp.route("/about/<path:path>")
def about_redirect(path: str):
    """Redirect legacy about routes to the new public about page."""
    target = f"/about/{path}" if path else "/about"
    return redirect(target)


@api_bp.route("/classify", methods=["POST"])
def classify():
    """Return simplified classification with severity and category volume context."""
    try:
        body = request.get_json(silent=True) or {}
        message = body.get("message")
        cleaned_message, error_message = validate_message_text(message or "")
        if error_message:
            return jsonify({"error": error_message}), 400

        model_service = getattr(current_app, "model_service", None)
        if model_service is None:
            return jsonify({"error": "Classification unavailable right now."}), 503

        prediction_result = process_prediction_result(model_service, cleaned_message)
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
                # Filter to disaster-related messages (related == 1) for consistency with other endpoints
                df, displayable_category_columns = _prepare_displayable_data(df, category_columns)
                if displayable_category_columns and df is not None and not df.empty:
                    sums = df[displayable_category_columns].fillna(0).sum()
                    category_volumes = sums.to_dict()
            except (DataServiceError, Exception) as data_error:
                _log_api_error("POST /api/classify (data_service)", data_error)
                # Volumes are supplementary; classification continues with empty volumes

        thresholds_map = model_service.get_thresholds_map()
        raw_probs = prediction_result.get("probabilities") or {}
        raw_labels = prediction_result.get("labels") or {}
        fixed_probs, fixed_labels = run_hierarchy_correction(raw_probs, thresholds_map)
        hierarchy_result = {
            **prediction_result,
            "probabilities": fixed_probs,
            "labels": fixed_labels,
        }
        payload = _build_simplified_classification(
            hierarchy_result, category_volumes, thresholds_map
        )

        debug_flag = (request.args.get("debug") or "").strip().lower()
        if debug_flag in {"1", "true", "yes", "on"}:
            payload["debug"] = {
                "thresholds": {
                    key: round(_safe_float_prob(value), 4)
                    for key, value in thresholds_map.items()
                },
                "raw": {
                    "probabilities": {
                        key: round(_safe_float_prob(value), 4)
                        for key, value in raw_probs.items()
                    },
                    "labels": {
                        key: _safe_label_value(value)
                        for key, value in raw_labels.items()
                    },
                },
                "fixed": {
                    "probabilities": {
                        key: round(_safe_float_prob(value), 4)
                        for key, value in fixed_probs.items()
                    },
                    "labels": {
                        key: _safe_label_value(value)
                        for key, value in fixed_labels.items()
                    },
                },
            }
        return jsonify(payload)
    except (ValueError, ModelServiceError) as error:
        _log_api_error("POST /api/classify", error)
        return jsonify({"error": "Classification unavailable right now."}), 503
    except Exception as error:
        _log_api_error("POST /api/classify", error)
        return jsonify({"error": "Classification unavailable right now."}), 500
