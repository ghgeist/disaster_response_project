"""
Home page routes for the Disaster Response application.
"""
import logging

from flask import Blueprint, abort, current_app, send_from_directory
import pandas as pd
import sqlalchemy.exc

from app.forms import MessageForm
from app.services.data_service import DataServiceError
from app.utils.formatting import format_request_context
from app.utils.route_helpers import render_home_with_visualizations
from app.visualizations import ChartGenerator

logger = logging.getLogger(__name__)

home_bp = Blueprint('home', __name__)


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

        return render_home_with_visualizations(form, data_service, chart_generator)

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
