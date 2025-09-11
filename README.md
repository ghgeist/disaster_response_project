![Alt text](images/image.png)

# Signal Storm: Leveraging Machine Learning to Identify Requests for Help During Natural Disasters

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#️-architecture)
- [Quick Start](#-quick-start)
- [Data](#-data)
- [Model Design](#-model-design)
- [Experimentation](#-experimentation)
- [Web Application](#-web-application)
- [Model Performance](#-model-performance)
- [Development](#-development)
- [Project Structure](#-project-structure)
- [Dependencies](#️-dependencies)
- [License](#-license)
- [Support](#-support)
- [Troubleshooting](#-troubleshooting)

## Project Overview

Signal Storm is a machine learning pipeline designed to classify emergency messages into 36 disaster-related categories, enabling rapid response coordination during natural disasters. The system processes text messages (primarily from social media and direct reports) and automatically categorizes them to help emergency response agencies prioritize and route assistance effectively.

The project includes:
- **Modular ML Pipeline**: Clean, professional architecture with separated concerns
- **Web Application**: Interactive Flask app for real-time message classification
- **Experiment Tracking**: Organized system for testing different sampling strategies
- **Comprehensive Evaluation**: Detailed metrics and model comparison tools 

## 🏗️ Architecture

The project follows a modern, modular architecture that demonstrates professional ML engineering practices:

```
src/disaster_classifier/          # Core ML package
├── data/                         # Data processing modules
│   ├── loader.py                # Database loading and ETL
│   ├── preprocessor.py          # Text tokenization and cleaning
│   ├── etl_pipeline.py          # Complete ETL workflow
│   └── column_definitions.py    # Data schema definitions
├── models/                       # Machine learning components
│   ├── pipeline.py              # ML pipeline creation and training
│   └── samplers.py              # Sampling strategies (SMOTE, ADASYN)
├── evaluation/                   # Model evaluation
│   └── metrics.py               # Comprehensive metrics and reporting
└── utils/                        # Configuration and utilities
    ├── config.py                # System configuration
    ├── io.py                    # File I/O operations
    ├── interaction.py           # User interaction utilities
    └── experiment_tracker.py    # Experiment management

scripts/                          # Professional training and testing interface
├── 01_test_sampling_strategies.py  # Sampling strategy testing
├── 02_test_hyperparameters.py     # Hyperparameter optimization
├── 03_create_experimental_model.py # Experimental model creation
├── 04_create_production_model.py  # Production model creation
├── 06_create_lightweight_model.py # Lightweight model creation
├── compare_models.py            # Model comparison tool
└── run_batch_experiments.py     # Batch experiment runner

experiments/                      # Organized experiment results
├── baseline_no_sampling_v1/
├── smote_conservative_v1/
├── adasyn_moderate_v1/
└── conservative_sampling_v1/

app/                              # Web application
├── app.py                       # Flask application
├── visualizations.py            # Visualization components
└── templates/                   # HTML templates
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.12.0 or higher
- **Virtual Environment**: Recommended (activate before proceeding)

### Virtual Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/disaster-response-project.git
   cd disaster-response-project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK resources** (handled automatically):
   - punkt tokenizer
   - stopwords corpus
   - wordnet corpus

### Data Setup

1. **Process raw data**:
   ```bash
   python data/process_data.py data/01_raw/disaster_messages.csv data/01_raw/disaster_categories.csv data/02_stg/stg_disaster_response.db
   ```

2. **Train a model**:
   ```bash
   # Create production model
   python scripts/04_create_production_model.py
   
   # Create lightweight model (recommended for deployment)
   python scripts/06_create_lightweight_model.py
   ```

3. **Run the web application**:
   ```bash
   python run.py
   ```
   Open your browser to `http://localhost:3000`

### Replit Deployment

The Flask application is configured for deployment on Replit:

1. **Import the project** into your Replit workspace
2. **Install dependencies** (Replit will automatically run `pip install -r requirements.txt`)
3. **Set environment variables** (if needed):
   - `GDRIVE_MODEL_ID`: Google Drive file ID for model download (optional)
4. **Run the application**: Click the "Run" button in Replit
5. **Access the app**: Use the provided Replit URL

**Note**: The app automatically downloads the model from Google Drive if not present locally, making it easy to deploy without large model files in the repository.

## 📊 Data

The system processes two main datasets:

### disaster_messages.csv
- **Content**: Emergency messages from various sources during disasters
- **Languages**: Primarily English (with Haitian Creole translations)
- **Genres**: Direct reports, news, and social media
- **Features**: Message text, original text, genre classification

### disaster_categories.csv
- **Content**: Binary classification labels for 36 disaster categories
- **Categories**: Include medical_help, water, food, shelter, infrastructure_related, etc.
- **Related Column**: Indicates disaster relevance (0=not related, 1=related, 2=ambiguous)

## 🤖 Model Design

The machine learning pipeline consists of three main stages:

### 1. Text Processing
- **Tokenization**: Custom NLTK-based tokenizer with lemmatization
- **Normalization**: Case normalization, URL replacement, punctuation removal
- **Stop Word Removal**: English stop words filtered out

### 2. Feature Engineering
- **Vectorization**: CountVectorizer with custom tokenizer
- **TF-IDF Transformation**: Term frequency-inverse document frequency weighting
- **N-gram Support**: Configurable unigram and bigram features

### 3. Multi-label Classification
- **Algorithm**: RandomForestClassifier with MultiOutputClassifier
- **Sampling Strategies**: Baseline, SMOTE, ADASYN, and conservative sampling
- **Hyperparameter Tuning**: GridSearchCV for optimization

## 🧪 Experimentation

The system supports organized experimentation with different sampling strategies:

### Available Experiments
- **baseline_no_sampling**: No class balancing applied
- **smote_conservative**: SMOTE with conservative parameters
- **adasyn_moderate**: ADASYN with moderate parameters  
- **conservative_sampling**: Very conservative SMOTE approach

### Running Experiments
```bash
# Test sampling strategies
python scripts/01_test_sampling_strategies.py data/02_stg/stg_disaster_response.db

# Test hyperparameters
python scripts/02_test_hyperparameters.py data/02_stg/stg_disaster_response.db

# Compare experiment results
python scripts/compare_models.py
```

### Experiment Tracking
Each experiment is automatically organized in the `experiments/` directory with:
- Model files and parameters
- Evaluation metrics and visualizations
- Configuration and results summaries
- Reproducible experiment names

## 🌐 Web Application

The Flask web application provides:

### Features
- **Real-time Classification**: Input messages and get instant category predictions
- **Data Visualization**: Interactive charts showing message distribution and categories
- **Model Performance**: Visual representation of model metrics
- **Responsive Design**: Bootstrap-based modern interface
- **Cloud Deployment**: Optimized for Replit deployment with automatic model downloading

### Usage
1. Navigate to the main page to see data visualizations
2. Enter a message in the classification interface
3. View predicted categories with confidence scores
4. Explore the dataset through interactive charts

### Deployment Options

#### Local Development
```bash
python run.py
```

#### Replit Deployment
The application is pre-configured for Replit deployment:
- **Automatic Model Download**: Downloads model from Google Drive if not present
- **Environment Variables**: Supports `GDRIVE_MODEL_ID` for model access
- **Port Configuration**: Automatically uses Replit's assigned port
- **Error Handling**: Robust error handling for cloud deployment scenarios

## 📈 Model Performance

The system evaluates models using comprehensive metrics:

### Key Metrics
- **Precision**: Accuracy of positive predictions per category
- **Recall**: Ability to find all positive instances per category
- **F1-Score**: Harmonic mean of precision and recall
- **Macro/Micro Averages**: Overall performance across categories

### Evaluation Approach
- **Multi-label Classification**: Handles overlapping categories
- **Class Imbalance**: Addresses skewed category distributions
- **Cross-validation**: Robust performance estimation
- **Statistical Significance**: Confidence intervals for metrics

## 🔧 Development

### Code Quality
- **Modular Design**: Single responsibility principle
- **Type Hints**: Comprehensive type annotations
- **Error Handling**: Robust exception management
- **Logging**: Detailed logging for debugging and monitoring
- **Documentation**: Comprehensive docstrings and comments

### Testing
```bash
# Validate project structure
python scripts/system_validation.py

# Run batch experiments
python scripts/run_batch_experiments.py

# Validate multilabel sampling
python scripts/validate_multilabel_sampling.py
```

### Contributing
1. Follow the established modular architecture
2. Maintain single responsibility for functions
3. Add comprehensive docstrings
4. Include error handling and logging
5. Update tests and documentation

## 📁 Project Structure

```
disaster_response_project/
├── src/disaster_classifier/     # Core ML package
├── scripts/                     # Training and utility scripts
├── experiments/                 # Experiment results
├── app/                         # Web application
├── data/                        # Data storage (raw, processed, results)
├── models/                      # Trained models and parameters
├── notebooks/                   # Jupyter notebooks for analysis
├── docs/                        # Documentation and guides
└── tests/                       # Unit tests
```

## 🛠️ Dependencies

### Core ML Libraries
- **scikit-learn**: Machine learning algorithms and utilities
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **nltk**: Natural language processing

### Web Application
- **Flask**: Web framework
- **Bootstrap**: Frontend styling
- **Plotly**: Interactive visualizations

### Data Management
- **SQLAlchemy**: Database operations
- **joblib**: Model serialization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Check the project documentation in the `docs/` directory
- Review the troubleshooting section below

## 🔧 Troubleshooting

### Common Issues

**Model not found error:**
```bash
# Ensure you've trained a model first
python scripts/04_create_production_model.py
```

**Database connection issues:**
```bash
# Verify database exists and is accessible
python data/process_data.py data/01_raw/disaster_messages.csv data/01_raw/disaster_categories.csv data/02_stg/stg_disaster_response.db
```

**Port already in use:**
```bash
# Use a different port
export PORT=3001
python run.py
```

**Missing dependencies:**
```bash
# Reinstall requirements
pip install -r requirements.txt
```
