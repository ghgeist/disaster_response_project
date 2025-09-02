"""
Routes for the Disaster Response application.
"""
import json
import logging
from flask import render_template, request, current_app, send_from_directory, abort
import plotly

from app.services import DataService, ModelService
from app.visualizations import ChartGenerator
from app.utils import validate_message_input, sanitize_input

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

            # Encode plotly graphs in JSON
            ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
            graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

            return render_template('master.html', ids=ids, graphJSON=graph_json)

        except Exception as e:
            logger.error(f"Error in index route: {e}")
            abort(500, description=f"Error preparing data for visualization: {e}")

    @app.route('/go')
    def go():
        """
        Handle user query and display model classification results.
        """
        try:
            # Get services from app context
            model_service = current_app.model_service
            
            # Get and validate user input
            query = request.args.get('query', '')
            query = sanitize_input(query)
            
            # Validate input
            is_valid, error_message = validate_message_input(query)
            if not is_valid:
                return render_template('error.html', message=error_message)

            # Use model to predict classification for query
            classification_results = model_service.predict(query)

            # Render results
            return render_template(
                'go.html',
                query=query,
                classification_result=classification_results
            )

        except Exception as e:
            logger.error(f"Error in go route: {e}")
            return render_template('error.html', message=f"Error processing query: {e}")

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
        return render_template('error.html', message="Page not found"), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        return render_template('error.html', message="Internal server error"), 500
