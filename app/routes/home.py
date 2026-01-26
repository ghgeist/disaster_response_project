"""
Home page routes for the Disaster Response application.
"""
import json
import logging
from flask import Blueprint, render_template, current_app, send_from_directory, abort
import plotly
import sqlalchemy.exc
import pandas as pd

from app.services.data_service import DataServiceError
from app.services.metrics_service import load_metric_frames, extract_perf_triplet
from app.visualizations import ChartGenerator
from app.utils.formatting import format_request_context
from app.forms import MessageForm

logger = logging.getLogger(__name__)

home_bp = Blueprint('home', __name__)


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


@home_bp.route('/favicon.ico')
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


@home_bp.route('/')
@home_bp.route('/index')
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
