"""
Routes for the Disaster Response application.
"""
import json
import logging
from pathlib import Path
from flask import render_template, request, current_app, send_from_directory, abort, flash, redirect, url_for
import plotly
import sqlalchemy.exc
import pandas as pd

from .services import (
    DataServiceError,
    ModelHealthMonitor,
    ModelServiceError,
    extract_perf_triplet,
    load_metric_frames,
)
from .visualizations import ChartGenerator
from .utils import format_request_context, sanitize_input, validate_message_input
from .forms import MessageForm
from .nltk_setup import get_nltk_status

# Import hierarchy functions
from disasterproject.hierarchy import apply_hierarchy, count_violations
from disasterproject.utils.config import (
    TAXONOMY,
    CRITICAL_LABELS,
    EXCLUDE_FROM_CONSTRAINTS,
    HIERARCHY_CRITICAL_THRESHOLD_REDUCTION
)

logger = logging.getLogger(__name__)


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


def _create_basic_visualizations(data_service, chart_generator):
    """
    Create basic genre and message type visualizations.
    
    Args:
        data_service: DataService instance
        chart_generator: ChartGenerator instance
        
    Returns:
        tuple: (graphs_list, descriptions_list)
    """
    df = data_service.get_data()
    
    # Create genre visualization
    genre_names, genre_related_counts = chart_generator.prepare_genre_data(df)
    genre_graph = chart_generator.create_genre_visual(genre_names, genre_related_counts)
    
    # Create message types visualization
    message_types_df = chart_generator.classify_message_types(df)
    message_type_graph = chart_generator.plot_message_types(message_types_df)
    
    graphs = [genre_graph, message_type_graph]
    descriptions = [
        "Direct messages dominate disaster communications. Bars show counts by source, stacked by disaster-related vs not. The predominance of direct messages underscores the need to triage individual cries for help.",
        "Among disaster-related direct messages, requests for aid are far more common than offers; direct reports are frequent. The model must reliably identify these requests."
    ]
    
    return graphs, descriptions


def _add_performance_visualization(graphs, descriptions):
    """
    Add performance visualization if data is available.
    
    Args:
        graphs: List of existing graphs
        descriptions: List of existing descriptions
        
    Returns:
        tuple: (updated_graphs, updated_descriptions)
    """
    context = format_request_context()
    try:
        base_df, opt_df = load_metric_frames()

        if base_df is not None and opt_df is not None:
            metrics, labels = extract_perf_triplet(base_df, opt_df)
            perf_graph = ChartGenerator.create_performance_visual(metrics, labels)
            graphs.append(perf_graph)
            descriptions.append(
                "Baseline (blue) vs Optimized (orange). Precision improves slightly; recall drops significantly. In disasters, missing real help messages is costly."
            )
        else:
            logger.warning("Performance CSVs missing; skipping performance chart%s", context)
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as perf_exc:
        logger.warning(
            "Skipping performance chart due to data issue%s: %s",
            context,
            perf_exc,
        )
    except Exception as perf_exc:
        logger.warning(
            "Skipping performance chart due to unexpected error%s: %s",
            context,
            perf_exc,
        )

    return graphs, descriptions


def _encode_graphs_to_json(graphs):
    """
    Encode plotly graphs to JSON format.
    
    Args:
        graphs: List of plotly graph objects
        
    Returns:
        tuple: (graph_json_string, ids_list)
    """
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    try:
        graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
        return graph_json, ids
    except (TypeError, ValueError) as json_error:
        logger.error("Error encoding graphs to JSON: %s", json_error)
        return "[]", []


def register_routes(app):
    """Register all application routes."""
    
    @app.route('/favicon.ico')
    def favicon():
        """Serve favicon with fallback options."""
        try:
            images_dir = current_app.config['IMAGES_DIR']
            ico_path = images_dir / 'favicon.ico'
            png_fallbacks = ['favicon.png', 'image.png']

            if ico_path.exists():
                return send_from_directory(images_dir, 'favicon.ico', mimetype='image/x-icon')

            for png_name in png_fallbacks:
                png_path = images_dir / png_name
                if png_path.exists():
                    return send_from_directory(images_dir, png_name, mimetype='image/png')

            abort(404)
            
        except (OSError, FileNotFoundError) as error:
            context = format_request_context()
            logger.error("Favicon access failed%s: %s", context, error)
            abort(404)
        except Exception:
            context = format_request_context()
            logger.exception("Unhandled favicon error%s", context)
            abort(404)

    @app.route('/')
    @app.route('/index')
    def index():
        """
        Main page displaying visualizations and message classification form.
        """
        try:
            # Create form instance
            form = MessageForm()
            
            
            # Get services from app context
            data_service = current_app.data_service
            chart_generator = ChartGenerator()
            
            # Create basic visualizations
            graphs, descriptions = _create_basic_visualizations(data_service, chart_generator)
            
            # Add performance visualization if available
            graphs, descriptions = _add_performance_visualization(graphs, descriptions)
            
            # Encode graphs to JSON
            graph_json, ids = _encode_graphs_to_json(graphs)

            return render_template('home.html', form=form, ids=ids, graphJSON=graph_json, descriptions=descriptions)

        except (sqlalchemy.exc.SQLAlchemyError, pd.errors.DatabaseError) as error:
            context = format_request_context()
            logger.error("Index rendering blocked by database error%s: %s", context, error)
            abort(500, description="Database unavailable.")
        except DataServiceError as error:
            context = format_request_context()
            logger.error("Index rendering blocked by data service error%s: %s", context, error)
            abort(500, description="Data service unavailable.")
        except (OSError, FileNotFoundError) as error:
            context = format_request_context()
            logger.error("Index rendering blocked by missing files%s: %s", context, error)
            abort(500, description="Required data missing.")
        except Exception:
            context = format_request_context()
            logger.exception("Unhandled index error%s", context)
            abort(500, description="Unexpected server error.")

    @app.route('/go', methods=['GET', 'POST'], strict_slashes=False)
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
                return redirect(url_for('index'))
            
            try:
                model_service = current_app.model_service
                query = sanitize_input(query)
                is_valid, error_message = validate_message_input(query)
                
                if not is_valid:
                    flash(error_message, 'error')
                    return redirect(url_for('index'))

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
            return redirect(url_for('index'))
        
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
                    return redirect(url_for('index'))
                except Exception:
                    context = format_request_context()
                    logger.exception("Unhandled POST /go error%s", context)
                    flash("An unexpected error occurred. Please try again.", 'error')
                    return redirect(url_for('index'))
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

    @app.route('/classify', methods=['GET', 'POST'], strict_slashes=False)
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
                return redirect(url_for('index'))

        else:  # POST request
            if not form.validate_on_submit():
                # Handle form validation errors
                for field, errors in form.errors.items():
                    for error in errors:
                        flash(f"{form[field].label.text}: {error}", 'error')
                return redirect(url_for('index'))

            query = sanitize_input(form.query.data)

        # Validate input
        is_valid, error_message = validate_message_input(query)
        if not is_valid:
            flash(error_message, 'error')
            return redirect(url_for('index'))

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
            return redirect(url_for('index'))
        except Exception:
            context = format_request_context()
            logger.exception("Unhandled /classify error%s", context)
            flash("An unexpected error occurred. Please try again.", 'error')
            return redirect(url_for('index'))

    @app.route('/health')
    def health_check():
        """
        Health check endpoint for monitoring with performance timing.
        """
        import time
        start_time = time.time()
        
        try:
            # Check if services are available
            data_service = current_app.data_service
            model_service = current_app.model_service
            
            # Test data service with timing
            data_start = time.time()
            df = data_service.get_data()
            data_time = (time.time() - data_start) * 1000
            data_healthy = len(df) > 0
            
            # Test model service with timing
            model_start = time.time()
            model = model_service.load_model()
            model_time = (time.time() - model_start) * 1000
            model_healthy = model is not None
            
            # Get NLTK status if available
            nltk_status = current_app.config.get('NLTK_SETUP_RESULTS', {})
            
            # Calculate total response time
            total_time = (time.time() - start_time) * 1000
            
            if data_healthy and model_healthy:
                response_data = {
                    'status': 'healthy',
                    'data_service': 'ok',
                    'model_service': 'ok',
                    'message_count': len(df),
                    'performance': {
                        'total_response_time_ms': round(total_time, 2),
                        'data_service_time_ms': round(data_time, 2),
                        'model_service_time_ms': round(model_time, 2)
                    }
                }
                
                # Add NLTK status if available
                if nltk_status:
                    response_data['nltk_status'] = {
                        'setup_success': nltk_status.get('success', False),
                        'setup_time_ms': nltk_status.get('setup_time_ms', 0),
                        'resources_loaded': len(nltk_status.get('resources_loaded', [])),
                        'resources_failed': len(nltk_status.get('resources_failed', []))
                    }
                
                return response_data, 200
            else:
                return {
                    'status': 'unhealthy',
                    'data_service': 'ok' if data_healthy else 'error',
                    'model_service': 'ok' if model_healthy else 'error',
                    'performance': {
                        'total_response_time_ms': round(total_time, 2),
                        'data_service_time_ms': round(data_time, 2),
                        'model_service_time_ms': round(model_time, 2)
                    }
                }, 503
                
        except (sqlalchemy.exc.SQLAlchemyError, pd.errors.DatabaseError) as error:
            context = format_request_context()
            logger.error("Health check database failure%s: %s", context, error)
            return {
                'status': 'unhealthy',
                'data_service': 'error',
                'model_service': 'unknown',
                'error': 'Database connection failed'
            }, 503
        except (OSError, FileNotFoundError, RuntimeError, DataServiceError, ModelServiceError) as error:
            context = format_request_context()
            logger.error("Health check service failure%s: %s", context, error)
            return {
                'status': 'unhealthy',
                'data_service': 'unknown',
                'model_service': 'error',
                'error': 'Service initialization failed'
            }, 503
        except Exception:
            context = format_request_context()
            logger.exception("Unhandled health check error%s", context)
            return {
                'status': 'unhealthy',
                'error': 'Unexpected system error'
            }, 503

    @app.route('/admin/model-health')
    def model_health_dashboard():
        """
        Model performance monitoring dashboard for admin users.
        """
        try:
            # Get services from app context
            model_service = getattr(current_app, 'model_service', None)
            
            # Initialize model health monitor with model service
            health_monitor = ModelHealthMonitor(model_service=model_service)
            
            # Get comprehensive health report
            health_report = health_monitor.get_comprehensive_health_report(model_service)
            
            return render_template(
                'model_health.html', 
                health_report=health_report,
                graphJSON="[]",  # Will be populated by JavaScript
                ids=[]
            )
            
        except Exception as error:
            context = format_request_context()
            logger.error("Model health dashboard failed%s: %s", context, error)
            return render_template(
                'error.html',
                message="Model health dashboard unavailable"
            ), 503

    @app.route('/api/model-health')
    def model_health_api():
        """
        API endpoint for model health data (for real-time updates).
        """
        try:
            # Get services from app context
            model_service = getattr(current_app, 'model_service', None)
            
            # Initialize model health monitor with model service
            health_monitor = ModelHealthMonitor(model_service=model_service)
            
            # Get comprehensive health report
            health_report = health_monitor.get_comprehensive_health_report(model_service)
            
            return health_report
            
        except Exception as error:
            context = format_request_context()
            logger.error("Model health API failed%s: %s", context, error)
            return {
                'error': str(error),
                'timestamp': pd.Timestamp.now().isoformat()
            }, 500

    @app.route('/api/performance-diagnostics')
    def performance_diagnostics():
        """
        API endpoint for performance diagnostics including NLTK and compatibility status.
        """
        try:
            import time
            start_time = time.time()
            
            # Get NLTK status
            nltk_status = get_nltk_status()
            
            
            # Get NLTK setup results from app config
            nltk_setup_results = current_app.config.get('NLTK_SETUP_RESULTS', {})
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            
            diagnostics = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'response_time_ms': round(response_time, 2),
                'nltk_status': nltk_status,
                'nltk_setup_results': nltk_setup_results,
                'performance_optimizations': {
                    'nltk_startup_optimization': 'enabled',
                    'per_request_downloads': 'disabled'
                }
            }
            
            return diagnostics
            
        except Exception as error:
            context = format_request_context()
            logger.error("Performance diagnostics API failed%s: %s", context, error)
            return {
                'error': str(error),
                'timestamp': pd.Timestamp.now().isoformat()
            }, 500

    @app.errorhandler(404)
    def not_found(_error):
        """Handle 404 errors."""
        return render_template('error.html', message="Page not found"), 404

    @app.errorhandler(500)
    def internal_error(_error):
        """Handle 500 errors."""
        return render_template('error.html', message="Internal server error"), 500
