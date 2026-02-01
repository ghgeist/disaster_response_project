"""
Health check and monitoring routes for the Disaster Response application.
"""
import logging
import time

import pandas as pd
import sqlalchemy.exc
from flask import Blueprint, current_app, render_template

from app.services.data_service import DataServiceError
from app.services.health_service import ModelHealthMonitor
from app.services.model_service import ModelServiceError
from app.utils.formatting import format_request_context
from app.utils.nltk_setup import get_nltk_status

logger = logging.getLogger(__name__)

health_bp = Blueprint('health', __name__)


@health_bp.route('/health')
def health_check():
    """
    Lightweight health check endpoint for deployment monitoring (e.g., Replit).
    Returns quickly without loading models or querying databases.
    """
    return {'status': 'ok'}, 200


@health_bp.route('/health/detailed')
def health_check_detailed():
    """
    Detailed health check endpoint for monitoring with performance timing.
    Checks data service, model service, and returns comprehensive diagnostics.
    """
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


@health_bp.route('/admin/model-health')
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


@health_bp.route('/api/model-health')
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


@health_bp.route('/api/performance-diagnostics')
def performance_diagnostics():
    """
    API endpoint for performance diagnostics including NLTK and compatibility status.
    """
    try:
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

