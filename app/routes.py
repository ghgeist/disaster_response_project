"""
Routes for the Disaster Response application.
"""
import json
import logging
from flask import render_template, request, current_app, send_from_directory, abort, flash, redirect, url_for
import plotly
import sqlalchemy.exc
import pandas as pd

from .services import load_metric_frames, extract_perf_triplet, ModelHealthMonitor
from .visualizations import ChartGenerator
from .utils import validate_message_input, sanitize_input
from .forms import MessageForm

logger = logging.getLogger(__name__)


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
            logger.warning("Performance CSVs missing; skipping performance chart.")
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as perf_exc:
        logger.warning("Skipping performance chart due to data issue: %s", perf_exc)
    except Exception as perf_exc:
        logger.warning("Skipping performance chart due to unexpected error: %s", perf_exc)
    
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
            
        except (OSError, FileNotFoundError) as e:
            logger.error("Error serving favicon - file system issue: %s", e)
            abort(404)
        except Exception as e:
            logger.error("Unexpected error serving favicon: %s", e)
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

            return render_template('master.html', form=form, ids=ids, graphJSON=graph_json, descriptions=descriptions)

        except (sqlalchemy.exc.SQLAlchemyError, pd.errors.DatabaseError) as e:
            logger.error("Database error in index route: %s", e)
            abort(500, description="Database connection error. Please try again later.")
        except (OSError, FileNotFoundError) as e:
            logger.error("File system error in index route: %s", e)
            abort(500, description="Required data files not found. Please contact administrator.")
        except Exception as e:
            logger.error("Unexpected error in index route: %s", e)
            abort(500, description="An unexpected error occurred. Please try again later.")

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

                classification_results = model_service.predict(query)
                flash('Message analyzed successfully!', 'success')

                return render_template(
                    'go.html',
                    form=form,
                    query=query,
                    classification_result=classification_results
                )
            except (ValueError, RuntimeError) as e:
                logger.error("Model prediction error in go route (GET): %s", e)
                flash("Error processing message. Please try again.", 'error')
            except Exception as e:
                logger.error("Unexpected error in go route (GET): %s", e)
                flash("An unexpected error occurred. Please try again.", 'error')
            return redirect(url_for('index'))
        
        # This point is only reached for POST requests.
        if form.validate_on_submit():
            query = sanitize_input(form.query.data)
            is_valid, error_message = validate_message_input(query)

            if is_valid:
                try:
                    model_service = current_app.model_service
                    classification_results = model_service.predict(query)
                    flash('Message analyzed successfully!', 'success')

                    return render_template(
                        'go.html',
                        form=form,
                        query=query,
                        classification_result=classification_results
                    )
                except (ValueError, RuntimeError) as e:
                    logger.exception("Model prediction failed in /go route (POST). See traceback:")
                    flash("Error processing message. Please try again.", 'error')
                    return redirect(url_for('index'))
                except Exception as e:
                    logger.exception("An unexpected error occurred in /go route (POST). See traceback:")
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

            return render_template('master.html', form=form, ids=ids, graphJSON=graph_json, descriptions=descriptions)
        except Exception as e:
            logger.error("Error re-rendering index page on form validation failure: %s", e)
            flash("An error occurred while processing your request.", 'error')
            return render_template('master.html', form=form, ids=[], graphJSON="[]", descriptions=[])

    @app.route('/health')
    def health_check():
        """
        Health check endpoint for monitoring.
        """
        try:
            # Check if services are available
            data_service = current_app.data_service
            model_service = current_app.model_service
            
            # Test data service
            df = data_service.get_data()
            data_healthy = len(df) > 0
            
            # Test model service
            model = model_service.load_model()
            model_healthy = model is not None
            
            if data_healthy and model_healthy:
                return {
                    'status': 'healthy',
                    'data_service': 'ok',
                    'model_service': 'ok',
                    'message_count': len(df)
                }, 200
            else:
                return {
                    'status': 'unhealthy',
                    'data_service': 'ok' if data_healthy else 'error',
                    'model_service': 'ok' if model_healthy else 'error'
                }, 503
                
        except (sqlalchemy.exc.SQLAlchemyError, pd.errors.DatabaseError) as e:
            logger.error("Database error in health check: %s", e)
            return {
                'status': 'unhealthy',
                'data_service': 'error',
                'model_service': 'unknown',
                'error': 'Database connection failed'
            }, 503
        except (OSError, FileNotFoundError, RuntimeError) as e:
            logger.error("Service error in health check: %s", e)
            return {
                'status': 'unhealthy',
                'data_service': 'unknown',
                'model_service': 'error',
                'error': 'Service initialization failed'
            }, 503
        except Exception as e:
            logger.error("Unexpected error in health check: %s", e)
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
            
        except Exception as e:
            logger.error(f"Error in model health dashboard: {e}")
            return render_template(
                'error.html', 
                message="Model health dashboard unavailable", 
                graphJSON="[]", 
                ids=[], 
                form=MessageForm()
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
            
        except Exception as e:
            logger.error(f"Error in model health API: {e}")
            return {
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat()
            }, 500

    @app.errorhandler(404)
    def not_found(_error):
        """Handle 404 errors."""
        form = MessageForm()
        return render_template('error.html', message="Page not found", graphJSON="[]", ids=[], form=form), 404

    @app.errorhandler(500)
    def internal_error(_error):
        """Handle 500 errors."""
        form = MessageForm()
        return render_template('error.html', message="Internal server error", graphJSON="[]", ids=[], form=form), 500
