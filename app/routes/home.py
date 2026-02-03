"""
Home page routes for the Disaster Response application.
"""
import logging

from flask import Blueprint, abort, current_app, redirect, send_from_directory

from app.utils.formatting import format_request_context

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
    Redirect root route to the React dashboard.
    """
    return redirect('/dashboard')


def _serve_spa():
    """Serve the React SPA shell for client-side routing."""
    static_folder = current_app.static_folder
    return send_from_directory(
        static_folder, "dashboard/index.html", mimetype="text/html"
    )


@home_bp.route("/dashboard")
@home_bp.route("/dashboard/", defaults={"path": ""})
@home_bp.route("/dashboard/<path:path>")
def dashboard(path: str | None = None):
    """Serve Storm Signal dashboard SPA (React app)."""
    return _serve_spa()


@home_bp.route("/production-model")
@home_bp.route("/production-model/", defaults={"path": ""})
@home_bp.route("/production-model/<path:path>")
def production_model(path: str | None = None):
    """Serve Model Information dashboard SPA (React app)."""
    return _serve_spa()


@home_bp.route("/about")
@home_bp.route("/about/", defaults={"path": ""})
@home_bp.route("/about/<path:path>")
def about(path: str | None = None):
    """Serve Storm Signal About page SPA (React app)."""
    return _serve_spa()
