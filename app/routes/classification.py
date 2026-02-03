"""
Message classification routes for the Disaster Response application.
"""
import json
import logging
from pathlib import Path

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from app.forms import MessageForm
from app.services.model_service import ModelServiceError
from app.utils.formatting import format_request_context
from app.utils.hierarchy_helpers import compute_violations
from app.utils.prediction_helpers import (
    format_predictions_for_display,
    process_prediction_result,
)
from app.utils.route_helpers import (
    handle_prediction_error,
    handle_validation_errors,
    render_home_with_visualizations,
)
from app.utils.validation import validate_message_text
from app.visualizations import ChartGenerator

# Import hierarchy functions
from disasterproject.hierarchy import apply_hierarchy
from disasterproject.utils.config import (
    CRITICAL_LABELS,
    EXCLUDE_FROM_CONSTRAINTS,
    HIERARCHY_CRITICAL_THRESHOLD_REDUCTION,
    TAXONOMY,
)

logger = logging.getLogger(__name__)

classification_bp = Blueprint('classification', __name__)


def _render_go_results(prediction_result: dict) -> str:
    """Render the results template for the legacy /go flow."""
    return render_template(
        "results.html",
        query=prediction_result["query"],
        sorted_predictions=prediction_result["sorted_predictions"],
        probabilities=prediction_result["probabilities"],
    )


def _render_home_with_form_errors(form: MessageForm):
    """Re-render the home page with form errors and refreshed charts."""
    try:
        handle_validation_errors(form)
        data_service = current_app.data_service
        chart_generator = ChartGenerator()
        return render_home_with_visualizations(form, data_service, chart_generator)
    except Exception as error:
        context = format_request_context()
        logger.error("Failed to re-render index after validation error%s: %s", context, error)
        flash("An error occurred while processing your request.", "error")
        return render_template(
            "home.html",
            form=form,
            ids=[],
            graphJSON="[]",
            descriptions=[],
        )


def _load_demo_metrics():
    """Load static demo metrics for hierarchy results (non-fatal if missing)."""
    try:
        metrics_path = Path(current_app.static_folder) / "demo_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _get_classify_query(form: MessageForm):
    """Return the query and an early response when validation fails."""
    if request.method == "GET":
        query = request.args.get("query", "")
        if not query:
            return None, redirect(url_for("home.index"))
        cleaned, error_message = validate_message_text(query)
        if error_message:
            flash(error_message, "error")
            return None, redirect(url_for("home.index"))
        return cleaned, None

    if not form.validate_on_submit():
        handle_validation_errors(form)
        return None, redirect(url_for("home.index"))

    return form.query.data, None


def _build_classify_response(prediction_result: dict, model_service):
    """Assemble the response payload for /classify."""
    raw_labels = prediction_result["labels"]
    raw_probabilities = prediction_result["probabilities"]

    service_thresholds = model_service.get_thresholds_map()
    thresholds = {
        label: service_thresholds.get(label, 0.5) for label in raw_probabilities.keys()
    }

    violations = compute_violations(
        raw_probabilities,
        TAXONOMY,
        EXCLUDE_FROM_CONSTRAINTS,
    )

    fixed_probabilities, fixed_labels = apply_hierarchy(
        probs=raw_probabilities,
        thresholds=thresholds,
        taxonomy=TAXONOMY,
        critical_labels=CRITICAL_LABELS,
        exclude=EXCLUDE_FROM_CONSTRAINTS,
        critical_threshold_reduction=HIERARCHY_CRITICAL_THRESHOLD_REDUCTION,
    )

    raw_payload = {"labels": raw_labels, "probabilities": raw_probabilities}
    fixed_payload = {"labels": fixed_labels, "probabilities": fixed_probabilities}

    return {
        "query": prediction_result["query"],
        "use_hierarchy": True,
        "raw": {
            "predictions": format_predictions_for_display(raw_payload),
            "probabilities": raw_probabilities,
            "labels": raw_labels,
        },
        "fixed": {
            "predictions": format_predictions_for_display(fixed_payload),
            "probabilities": fixed_probabilities,
            "labels": fixed_labels,
        },
        "violations": violations,
        "metrics": _load_demo_metrics(),
    }


def _handle_go_get():
    """Handle legacy GET requests to /go for backward compatibility."""
    query = request.args.get("query", "")
    if not query:
        return redirect(url_for("home.index"))
    cleaned, error_message = validate_message_text(query)
    if error_message:
        flash(error_message, "error")
        return redirect(url_for("home.index"))

    try:
        model_service = current_app.model_service
        prediction_result = process_prediction_result(model_service, cleaned)
        if not prediction_result["is_valid"]:
            flash(prediction_result["error_message"], "error")
            return redirect(url_for("home.index"))
        return _render_go_results(prediction_result)
    except (ValueError, ModelServiceError) as error:
        context = format_request_context()
        handle_prediction_error(
            error,
            {
                "logger": logger,
                "request_context": context,
                "log_message": "Prediction failure on GET /go%s: %s",
            },
        )
    except Exception as error:
        context = format_request_context()
        handle_prediction_error(
            error,
            {
                "logger": logger,
                "request_context": context,
                "log_message": "Unhandled GET /go error%s",
                "user_message": "An unexpected error occurred. Please try again.",
                "log_exception": True,
            },
        )
    return redirect(url_for("home.index"))


def _handle_go_post(form: MessageForm):
    """Handle POST form submissions for the /go endpoint."""
    if form.validate_on_submit():
        try:
            model_service = current_app.model_service
            prediction_result = process_prediction_result(model_service, form.query.data)
            if not prediction_result["is_valid"]:
                form.query.errors = (prediction_result["error_message"],)
                return _render_home_with_form_errors(form)
            return _render_go_results(prediction_result)
        except (ValueError, ModelServiceError) as error:
            context = format_request_context()
            handle_prediction_error(
                error,
                {
                    "logger": logger,
                    "request_context": context,
                    "log_message": "Prediction failure on POST /go%s: %s",
                },
            )
            return redirect(url_for("home.index"))
        except Exception as error:
            context = format_request_context()
            handle_prediction_error(
                error,
                {
                    "logger": logger,
                    "request_context": context,
                    "log_message": "Unhandled POST /go error%s",
                    "user_message": "An unexpected error occurred. Please try again.",
                    "log_exception": True,
                },
            )
            return redirect(url_for("home.index"))

    return _render_home_with_form_errors(form)


def _is_json_request() -> bool:
    """Return True when the caller requests a JSON response."""
    return (
        request.headers.get("Content-Type") == "application/json"
        or request.args.get("format") == "json"
    )


def _handle_classify_prediction(query: str):
    """Run prediction + hierarchy processing for the /classify endpoint."""
    try:
        model_service = current_app.model_service
        prediction_result = process_prediction_result(
            model_service,
            query,
            sanitize=request.method == "POST",
        )

        if not prediction_result["is_valid"]:
            flash(prediction_result["error_message"], "error")
            return redirect(url_for("home.index"))

        response_data = _build_classify_response(prediction_result, model_service)

        if _is_json_request():
            return response_data

        return render_template("results.html", **response_data)
    except (ValueError, ModelServiceError) as error:
        context = format_request_context()
        handle_prediction_error(
            error,
            {
                "logger": logger,
                "request_context": context,
                "log_message": "Prediction failure on /classify%s: %s",
            },
        )
    except Exception as error:
        context = format_request_context()
        handle_prediction_error(
            error,
            {
                "logger": logger,
                "request_context": context,
                "log_message": "Unhandled /classify error%s",
                "user_message": "An unexpected error occurred. Please try again.",
                "log_exception": True,
            },
        )
    return redirect(url_for("home.index"))


@classification_bp.route('/go', methods=['GET', 'POST'], strict_slashes=False)
def go():
    """
    Legacy/simple classification endpoint for the main UI and smoke tests.

    - GET supports the legacy query parameter for backward compatibility.
    - POST handles form submissions and renders results without hierarchy comparisons.
    - /classify remains for the hierarchy demo and JSON-friendly responses.
    """
    form = MessageForm()
    if request.method == "GET":
        return _handle_go_get()

    return _handle_go_post(form)


@classification_bp.route('/classify', methods=['GET', 'POST'], strict_slashes=False)
def classify():
    """
    Hierarchy-aware classification endpoint for the demo UI and API callers.

    Supports raw vs hierarchy-fixed results and can return JSON for API use.
    """
    form = MessageForm()
    query, early_response = _get_classify_query(form)
    if early_response is not None:
        return early_response

    return _handle_classify_prediction(query)
