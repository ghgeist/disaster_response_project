# Storm Signal: Disaster Response ML Pipeline

## Overview

Storm Signal is a machine learning pipeline that classifies emergency messages into 36 disaster-related categories to enable rapid response coordination during natural disasters. The system includes a modular ML pipeline with separated concerns, an interactive Flask web application for real-time message classification, organized experiment tracking, and comprehensive evaluation tools. Built with professional ML engineering practices, it demonstrates clean architecture with a focus on scalability and maintainability.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Flask Web Application**: Built with a clean, modular design using Flask with Jinja2 templates
- **Dark Theme UI**: Modern design using Tailwind CSS with responsive layout and accessibility features
- **Interactive Visualizations**: Plotly.js charts for data distribution analysis and performance metrics
- **Form Handling**: Flask-WTF for secure form processing with CSRF protection
- **Real-time Classification**: AJAX-powered message classification with confidence scoring

### Backend Architecture
- **Modular Package Structure**: Core functionality organized in `src/disasterproject/` with clear separation of concerns
- **Data Processing**: ETL pipeline with SQLite database storage and preprocessing modules
- **ML Pipeline**: scikit-learn based pipeline with RandomForest and LogisticRegression options
- **Service Layer**: Dedicated services for data access, model management, and health monitoring
- **Standardized Model Naming**: Semantic versioning with format `{domain}_{algorithm}_{version}_{environment}_{date}.pkl`

### Data Storage Solutions
- **SQLite Database**: Structured storage in `data/02_stg/stg_disaster_response.db` for processed messages
- **CSV Data Sources**: Raw data ingestion from disaster_messages.csv and disaster_categories.csv
- **Model Artifacts**: Versioned model files with metadata, parameters, and training logs
- **Experiment Tracking**: Organized results storage with performance metrics and comparison tools

### Authentication and Authorization Mechanisms
- **CSRF Protection**: Flask-WTF CSRF tokens for form security
- **Input Validation**: Comprehensive form validation with HTML tag filtering and length limits
- **Session Management**: Flask session handling with secure cookie configuration
- **Error Handling**: Structured error pages with user-friendly messaging

### Deployment Strategy
- **Hybrid Model Deployment**: Google Drive primary storage with local fallback for development
- **Module Compatibility**: Runtime path patching to handle legacy model dependencies
- **Environment-based Configuration**: Separate configs for development, staging, and production
- **Health Monitoring**: Model performance dashboard with real-time metrics

## External Dependencies

### Third-party Services
- **Google Drive**: Production model storage and distribution (file ID: 1s_sBXnUdJ-rWm4-YEsDixHCbxBca-oXh)

### APIs and Libraries
- **Machine Learning**: scikit-learn, joblib, imblearn for model training and sampling strategies
- **Natural Language Processing**: NLTK for tokenization and text preprocessing
- **Web Framework**: Flask with Flask-WTF for secure web application development
- **Data Processing**: pandas, numpy, SQLAlchemy for data manipulation and database operations
- **Visualization**: Plotly.js for interactive charts and data visualization
- **Frontend**: Tailwind CSS, jQuery for responsive UI and interactivity

### Development Tools
- **Testing**: pytest for unit testing and smoke tests
- **Code Quality**: pylint for code analysis and standards enforcement
- **Package Management**: setuptools for installable package structure
- **Environment**: Python 3.12+ with virtual environment support

### Database Dependencies
- **SQLite**: Primary database engine for message storage and retrieval
- **SQLAlchemy**: ORM for database operations and query management