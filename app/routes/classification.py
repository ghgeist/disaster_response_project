"""
Message classification routes for the Disaster Response application.
"""
import json
import logging
from pathlib import Path
from flask import Blueprint, render_template, request, current_app, flash, redirect, url_for

from app.services.model_service import ModelServiceError
from app.utils.validation import sanitize_input, validate_message_input
from app.utils.formatting import format_request_context
from app.forms import MessageForm

# Import hierarchy functions
from disasterproject.hierarchy import apply_hierarchy
from disasterproject.utils.config import (
    TAXONOMY,
    CRITICAL_LABELS,
    EXCLUDE_FROM_CONSTRAINTS,
    HIERARCHY_CRITICAL_THRESHOLD_REDUCTION
)

logger = logging.getLogger(__name__)

classification_bp = Blueprint('classification', __name__)


def compute_violations(probs, taxonomy, exclude_set):
    """
    Compute parent < child violations for display in the diff table.

    Args:
        probs: Dictionary mapping label names to probabilities
        taxonomy: Dictionary mapping parent labels to list of child labels
        exclude_set: Set of labels to exclude from violation checks

    Returns:
        List of violation dictionaries with parent, child, parent_prob, child_prob
    """
    violations = []

    for parent, children in taxonomy.items():
        if parent == "related":
            continue  # Skip 'related' group as it doesn't use probability constraints

        if parent in exclude_set:
            continue

        # Find valid children (present in probs and not excluded)
        valid_children = [child for child in children
                         if child in probs and child not in exclude_set]

        if not valid_children or parent not in probs:
            continue

        # Find violations where child > parent
        for child in valid_children:
            if probs[child] > probs[parent]:
                violations.append({
                    'parent': parent,
                    'child': child,
                    'parent_prob': probs[parent],
                    'child_prob': probs[child]
                })

    return violations


@classification_bp.route('/go', methods=['GET', 'POST'], strict_slashes=False)
def go():
    """
    Handle user query and display model classification results.
    - POST: Validates form data and shows classification. On failure, re-renders
      the main page with errors and existing visualizations.
    - GET: Supports legacy requests with a 'query' parameter for backward compatibility.
    """
    form = MessageForm()
    
    # Handle legacy GET requests for backward compatibility
    if request.method == 'GET':
        query = request.args.get('query', '')
        if not query:
            return redirect(url_for('home.index'))
        
        try:
            model_service = current_app.model_service
            query = sanitize_input(query)
            is_valid, error_message = validate_message_input(query)
            
            if not is_valid:
                flash(error_message, 'error')
                return redirect(url_for('home.index'))

            prediction = model_service.predict(query)
            classification_results = prediction.get('labels', {})
            probabilities = prediction.get('probabilities', {})

            # Synthesize probabilities from labels when predict_proba unavailable
            if not probabilities and classification_results:
                probabilities = {k: 1.0 if v == 1 else 0.0 for k, v in classification_results.items()}

            # Create a unified list of predictions with their confidence
            # Exclude 'related' category from display as it's a meta-category indicating disaster relevance
            predictions = []
            for category, label in classification_results.items():
                if label == 1 and category != 'related':
                    predictions.append({
                        "category": category,
                        "confidence": probabilities.get(category, 0.0)  # Fallback to 0.0 if no probability
                    })
            
            # Sort predictions by confidence in descending order
            sorted_predictions = sorted(predictions, key=lambda p: p['confidence'], reverse=True)

            return render_template(
                'results.html',
                query=query,
                sorted_predictions=sorted_predictions,
                probabilities=probabilities
            )
        except (ValueError, ModelServiceError) as error:
            context = format_request_context()
            logger.error("Prediction failure on GET /go%s: %s", context, error)
            flash("Error processing message. Please try again.", 'error')
        except Exception:
            context = format_request_context()
            logger.exception("Unhandled GET /go error%s", context)
            flash("An unexpected error occurred. Please try again.", 'error')
        return redirect(url_for('home.index'))
    
    # This point is only reached for POST requests.
    if form.validate_on_submit():
        query = sanitize_input(form.query.data)
        is_valid, error_message = validate_message_input(query)

        if is_valid:
            try:
                model_service = current_app.model_service
                prediction = model_service.predict(query)
                classification_results = prediction.get('labels', {})
                probabilities = prediction.get('probabilities', {})

                # Synthesize probabilities from labels when predict_proba unavailable
                if not probabilities and classification_results:
                    probabilities = {k: 1.0 if v == 1 else 0.0 for k, v in classification_results.items()}
                
                # Create a unified list of predictions with their confidence
                # Exclude 'related' category from display as it's a meta-category indicating disaster relevance
                predictions = []
                for category, label in classification_results.items():
                    if label == 1 and category != 'related':
                        predictions.append({
                            "category": category,
                            "confidence": probabilities.get(category, 0.0)  # Fallback to 0.0 if no probability
                        })
                
                # Sort predictions by confidence in descending order
                sorted_predictions = sorted(predictions, key=lambda p: p['confidence'], reverse=True)

                return render_template(
                    'results.html',
                    query=query,
                    sorted_predictions=sorted_predictions,
                    probabilities=probabilities
                )
            except (ValueError, ModelServiceError) as error:
                context = format_request_context()
                logger.error("Prediction failure on POST /go%s: %s", context, error)
                flash("Error processing message. Please try again.", 'error')
                return redirect(url_for('home.index'))
            except Exception:
                context = format_request_context()
                logger.exception("Unhandled POST /go error%s", context)
                flash("An unexpected error occurred. Please try again.", 'error')
                return redirect(url_for('home.index'))
        else:
            # Custom validation failed. Add the error to the form's error list
            # so it can be displayed to the user on the re-rendered main page.
            form.query.errors = (error_message,)

    # Fall-through for POST requests with validation errors (either from WTForms
    # or our custom validation). Re-render the main page with errors.
    # We re-render the main page with the form to show errors and preserve input.
    try:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"{form[field].label.text}: {error}", 'error')
        
        # To re-render the main page, we need to regenerate the visualizations.
        from app.visualizations import ChartGenerator
        from app.routes.home import _create_basic_visualizations, _add_performance_visualization, _encode_graphs_to_json
        
        data_service = current_app.data_service
        chart_generator = ChartGenerator()
        graphs, descriptions = _create_basic_visualizations(data_service, chart_generator)
        graphs, descriptions = _add_performance_visualization(graphs, descriptions)
        graph_json, ids = _encode_graphs_to_json(graphs)

        return render_template('home.html', form=form, ids=ids, graphJSON=graph_json, descriptions=descriptions)
    except Exception as error:
        context = format_request_context()
        logger.error("Failed to re-render index after validation error%s: %s", context, error)
        flash("An error occurred while processing your request.", 'error')
        return render_template('home.html', form=form, ids=[], graphJSON="[]", descriptions=[])


@classification_bp.route('/classify', methods=['GET', 'POST'], strict_slashes=False)
def classify():
    """
    Classify messages with optional hierarchy processing.
    Supports both raw predictions and hierarchy-fixed results.
    """
    form = MessageForm()

    # Handle GET requests (for URL parameters or direct access)
    if request.method == 'GET':
        query = request.args.get('query', '')
        if not query:
            return redirect(url_for('home.index'))

    else:  # POST request
        if not form.validate_on_submit():
            # Handle form validation errors
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f"{form[field].label.text}: {error}", 'error')
            return redirect(url_for('home.index'))

        query = sanitize_input(form.query.data)

    # Validate input
    is_valid, error_message = validate_message_input(query)
    if not is_valid:
        flash(error_message, 'error')
        return redirect(url_for('home.index'))

    try:
        model_service = current_app.model_service

        # Get raw predictions from model
        prediction = model_service.predict(query)
        raw_labels = prediction.get('labels', {})
        raw_probabilities = prediction.get('probabilities', {})

        # Synthesize probabilities from labels when predict_proba unavailable
        if not raw_probabilities and raw_labels:
            raw_probabilities = {k: 1.0 if v == 1 else 0.0 for k, v in raw_labels.items()}

        # Get thresholds from model service (includes optimized thresholds if available)
        # Fallback to 0.5 for any missing labels
        service_thresholds = model_service.get_thresholds_map()
        thresholds = {
            label: service_thresholds.get(label, 0.5) 
            for label in raw_probabilities.keys()
        }

        # Compute violations in raw predictions
        violations = compute_violations(raw_probabilities, TAXONOMY, EXCLUDE_FROM_CONSTRAINTS)

        # Always apply hierarchy processing
        fixed_probabilities, fixed_labels = apply_hierarchy(
            probs=raw_probabilities,
            thresholds=thresholds,
            taxonomy=TAXONOMY,
            critical_labels=CRITICAL_LABELS,
            exclude=EXCLUDE_FROM_CONSTRAINTS,
            critical_threshold_reduction=HIERARCHY_CRITICAL_THRESHOLD_REDUCTION
        )

        # Create prediction lists for display (exclude 'related')
        raw_predictions = []
        fixed_predictions = []

        for category, label in raw_labels.items():
            if label == 1 and category != 'related':
                raw_predictions.append({
                    "category": category,
                    "confidence": raw_probabilities.get(category, 0.0)
                })

        for category, label in fixed_labels.items():
            if label == 1 and category != 'related':
                fixed_predictions.append({
                    "category": category,
                    "confidence": fixed_probabilities.get(category, 0.0)
                })

        # Sort predictions by confidence
        raw_predictions.sort(key=lambda p: p['confidence'], reverse=True)
        fixed_predictions.sort(key=lambda p: p['confidence'], reverse=True)

        # Load static demo metrics (non-fatal if missing)
        metrics = None
        try:
            metrics_path = Path(current_app.static_folder) / "demo_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
        except Exception as _:
            metrics = None

        # Prepare response data
        response_data = {
            'query': query,
            'use_hierarchy': True, # Always true now
            'raw': {
                'predictions': raw_predictions,
                'probabilities': raw_probabilities,
                'labels': raw_labels
            },
            'fixed': {
                'predictions': fixed_predictions,
                'probabilities': fixed_probabilities,
                'labels': fixed_labels
            },
            'violations': violations,
            'metrics': metrics
        }

        # For AJAX requests, return JSON
        if request.headers.get('Content-Type') == 'application/json' or request.args.get('format') == 'json':
            return response_data

        # For regular requests, render template
        return render_template(
            'results.html',
            **response_data
        )

    except (ValueError, ModelServiceError) as error:
        context = format_request_context()
        logger.error("Prediction failure on /classify%s: %s", context, error)
        flash("Error processing message. Please try again.", 'error')
        return redirect(url_for('home.index'))
    except Exception:
        context = format_request_context()
        logger.exception("Unhandled /classify error%s", context)
        flash("An unexpected error occurred. Please try again.", 'error')
        return redirect(url_for('home.index'))
