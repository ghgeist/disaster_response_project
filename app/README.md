# Disaster Response Flask Application

A clean, scalable Flask application for disaster response message classification. Perfect for portfolio projects with room for expansion.

## Architecture

```
app/
├── app.py              # Main Flask application
├── routes.py           # All application routes
├── services.py         # Data and model services
├── utils.py            # Utility functions
├── visualizations.py   # Chart generation
├── config.py           # Configuration
└── templates/          # Jinja2 templates
    ├── base.html       # Base template with common layout
    ├── home.html       # Main page with form and visualizations
    ├── results.html    # Classification results page
    ├── error.html      # Error page
    └── model_health.html # Model health dashboard
```

## Features

- **Clean Architecture**: Modular design that's easy to understand and extend
- **Data Visualization**: Interactive Plotly charts showing message distribution
- **ML Classification**: Predicts disaster categories for input messages
- **Model Management**: Automatic model downloading from Google Drive
- **Input Validation**: Secure input handling and validation
- **Health Monitoring**: Health check endpoint for production monitoring
- **Error Handling**: Comprehensive error handling with user-friendly messages

## Quick Start

1. **Set Environment Variables** (optional):
   ```bash
   export FLASK_ENV=development
   export GDRIVE_MODEL_ID=your_google_drive_file_id
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Visit**: http://localhost:5000

## Configuration

The app uses environment-based configuration:

- `FLASK_ENV`: Set to 'development' for debug mode
- `SECRET_KEY`: Flask secret key (auto-generated for development)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 5000)
- `GDRIVE_MODEL_ID`: Google Drive file ID for model download
- `LOG_LEVEL`: Logging level (INFO/DEBUG/WARNING/ERROR)

## API Endpoints

- `GET /`: Main page with visualizations and classification form
- `GET /go`: Message classification results
- `GET /health`: Lightweight health check endpoint (for deployment monitoring, e.g., Replit)
- `GET /health/detailed`: Detailed health check with service diagnostics and performance metrics
- `GET /favicon.ico`: Application favicon

## Adding New Features

This structure makes it easy to add new functionality:

1. **New Routes**: Add to `routes.py`
2. **New Services**: Add to `services.py` or create new service modules
3. **New Utilities**: Add to `utils.py`
4. **New Templates**: Add to `templates/` directory
5. **Configuration**: Update `config.py`

## Benefits

- **Portfolio Ready**: Clean, professional code structure
- **Scalable**: Easy to add new features and functionality
- **Maintainable**: Clear separation of concerns
- **Deployable**: Ready for production deployment
- **Testable**: Modular design supports easy testing
- **Documented**: Well-documented code and architecture

This simplified structure removes over-engineering while maintaining professional standards and scalability for your portfolio project.