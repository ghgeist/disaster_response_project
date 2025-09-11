"""
Routes for the Disaster Response application.
"""
import json
import logging
from flask import render_template, request, current_app, send_from_directory, abort, flash, redirect, url_for
import plotly

from .services import DataService, ModelService, load_metric_frames, extract_perf_triplet
from .visualizations import ChartGenerator
from .utils import validate_message_input, sanitize_input
from .forms import MessageForm

logger = logging.getLogger(__name__)


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
            
        except Exception as e:
            logger.error(f"Error serving favicon: {e}")
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
            
            # Load data
            df = data_service.get_data()
            
            # Create visualizations
            genre_names, genre_related_counts = chart_generator.prepare_genre_data(df)
            genre_graph = chart_generator.create_genre_visual(genre_names, genre_related_counts)

            message_types_df = chart_generator.classify_message_types(df)
            message_type_graph = chart_generator.plot_message_types(message_types_df)

            graphs = [genre_graph, message_type_graph]

            # Performance Deep Dive chart (best-effort; do not crash if missing)
            descriptions = []
            try:
                base_df, opt_df = load_metric_frames()
                if base_df is not None and opt_df is not None:
                    metrics, labels = extract_perf_triplet(base_df, opt_df)
                    perf_graph = ChartGenerator.create_performance_visual(metrics, labels)
                    graphs.append(perf_graph)
                    # Descriptions aligned to graphs (index-based)
                    descriptions = [
                        "Direct messages dominate disaster communications. Bars show counts by source, stacked by disaster-related vs not. The predominance of direct messages underscores the need to triage individual cries for help.",
                        "Among disaster-related direct messages, requests for aid are far more common than offers; direct reports are frequent. The model must reliably identify these requests.",
                        "Baseline (blue) vs Optimized (orange). Precision improves slightly; recall drops significantly. In disasters, missing real help messages is costly.",
                    ]
                else:
                    logger.warning("Performance CSVs missing; skipping performance chart.")
            except Exception as perf_exc:
                logger.warning(f"Skipping performance chart due to error: {perf_exc}")

            # Encode plotly graphs in JSON
            ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
            try:
                graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
            except (TypeError, ValueError) as json_error:
                logger.error(f"Error encoding graphs to JSON: {json_error}")
                # Fallback: create empty graphs array
                graph_json = "[]"
                ids = []

            return render_template('master.html', form=form, ids=ids, graphJSON=graph_json, descriptions=descriptions)

        except Exception as e:
            logger.error(f"Error in index route: {e}")
            abort(500, description=f"Error preparing data for visualization: {e}")

    @app.route('/go', methods=['GET', 'POST'])
    def go():
        """
        Handle user query and display model classification results.
        """
        form = MessageForm()
        
        # Handle GET requests (backward compatibility)
        if request.method == 'GET':
            query = request.args.get('query', '')
            if query:
                # Redirect to POST with flash message for better UX
                flash('Please use the form below to analyze messages.', 'info')
                return redirect(url_for('index'))
            else:
                return redirect(url_for('index'))
        
        # Handle POST requests with form validation
        if form.validate_on_submit():
            try:
                # Get services from app context
                model_service = current_app.model_service
                
                # Get and sanitize user input
                query = sanitize_input(form.query.data)
                
                # Additional validation (redundant but safe)
                is_valid, error_message = validate_message_input(query)
                if not is_valid:
                    flash(error_message, 'error')
                    return render_template('master.html', form=form, ids=[], graphJSON="[]", descriptions=[])

                # Use model to predict classification for query
                classification_results = model_service.predict(query)
                
                # Flash success message
                flash('Message analyzed successfully!', 'success')

                # Render results
                return render_template(
                    'go.html',
                    query=query,
                    classification_result=classification_results,
                    graphJSON="[]",  # Empty graphs array for go.html
                    ids=[]           # Empty ids array for go.html
                )

            except Exception as e:
                logger.error(f"Error in go route: {e}")
                flash(f"Error processing message: {e}", 'error')
                return render_template('master.html', form=form, ids=[], graphJSON="[]", descriptions=[])
        else:
            # Form validation failed - show errors
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f"{form[field].label.text}: {error}", 'error')
            
            # Return to index with form errors
            return redirect(url_for('index'))

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
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }, 503

    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return render_template('error.html', message="Page not found", graphJSON="[]", ids=[]), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        return render_template('error.html', message="Internal server error", graphJSON="[]", ids=[]), 500
