"""
Reusable route helpers for consistent UI behavior.
"""
from typing import Any, Mapping

from flask import flash, render_template

from app.utils.visualization_helpers import (
    _add_performance_visualization,
    _create_basic_visualizations,
    _encode_graphs_to_json,
)


def handle_validation_errors(form) -> None:
    """
    Flash WTForms validation errors in a consistent format.

    Args:
        form: WTForms form instance with errors populated.
    """
    for field, errors in form.errors.items():
        for error in errors:
            flash(f"{form[field].label.text}: {error}", "error")


def render_home_with_visualizations(form, data_service, chart_generator):
    """
    Render the home page with the standard visualization payload.

    Args:
        form: Message form instance.
        data_service: DataService instance.
        chart_generator: ChartGenerator instance.

    Returns:
        Flask response rendering home.html.
    """
    graphs, descriptions = _create_basic_visualizations(data_service, chart_generator)
    graphs, descriptions = _add_performance_visualization(graphs, descriptions)
    graph_json, ids = _encode_graphs_to_json(graphs)

    return render_template(
        "home.html",
        form=form,
        ids=ids,
        graphJSON=graph_json,
        descriptions=descriptions,
    )


def handle_prediction_error(error: Exception, context: Mapping[str, Any]) -> None:
    """
    Standardize prediction error logging and user messaging.

    Args:
        error: Exception raised during prediction.
        context: Dict containing logger and logging/message configuration.
    """
    logger = context.get("logger")
    request_context = context.get("request_context", "")
    log_message = context.get("log_message", "Prediction failure%s: %s")
    user_message = context.get(
        "user_message",
        "Error processing message. Please try again.",
    )
    flash_category = context.get("flash_category", "error")
    log_exception = context.get("log_exception", False)

    if logger:
        if log_exception:
            logger.exception(log_message, request_context)
        else:
            logger.error(log_message, request_context, error)

    flash(user_message, flash_category)
