![Alt text](images/image.png)

# Signal Storm: Leveraging Machine Learning to Identify Requests for Help During Natural Disasters

## Table of Contents
- [Highlights](#highlights)
- [Project Overview](#project-overview)
- [Architecture](#️-architecture)
- [Quick Start](#-quick-start)
- [Data](#-data)
- [Model Design](#-model-design)
- [Experimentation](#-experimentation)
- [Web Application](#-web-application)
- [Model Performance](#-model-performance)
- [Development](#-development)
- [Dev hygiene](#dev-hygiene)
- [Project Structure](#-project-structure)
- [Dependencies](#️-dependencies)
- [License](#-license)
- [Support](#-support)
- [Troubleshooting](#-troubleshooting)

## Highlights

- Model size reduced ~1000× enabling lightweight deployments.
- Load time dropped ~99% through on-demand initialization.
- Critical-label recall improved via targeted retraining.
- Hybrid deployment with Google Drive fallback and clear model naming.

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
src/disasterproject/          # Core ML package
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
    ├── json_io.py               # JSON I/O utilities
    ├── interaction.py           # User interaction utilities
    └── experiment_tracker.py    # Experiment management

scripts/                          # Professional training and testing interface
├── 01_test_sampling_strategies.py  # Sampling strategy testing
├── 02_test_hyperparameters.py     # Hyperparameter optimization
├── 03_create_experimental_model.py # Experimental model creation
├── 04_create_production_model.py  # Production model creation
├── compare_models.py            # Model comparison tool
├── compare_child_alone.py       # Child alone label analysis
├── evaluate_hierarchy.py        # Hierarchy constraint evaluation
├── optimize_hierarchy_threshold_reduction.py  # Optimize hierarchy post-processing parameter
├── optimize_per_category_thresholds.py        # Optimize individual category thresholds
├── promote_model.py             # Model promotion utility
├── test_experimental_model.py   # Experimental model testing
├── validate_multilabel_sampling.py # Multilabel sampling validation
├── system_validation.py         # System validation checks
├── deployment_health_check.py   # Deployment health verification
└── run_batch_experiments.py     # Batch experiment runner

experiments/                      # Organized experiment results
└── results/                      # Dated CSVs (e.g., 2025-09-13_lightweight_metrics.csv)

app/                              # Web application
├── app.py                       # Flask application factory
├── visualizations.py            # Visualization components
└── templates/                   # HTML templates

run.py                           # Application entry point
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.12.0 or higher
- **Virtual Environment**: Required for local development, **not needed** when SSH'd into Replit

### Virtual Environment Setup

**Note**: Virtual environment is required for local development but **not needed** when SSH'd into Replit. The project automatically detects the environment (using `REPLIT_DB_URL` or `REPL_ID` environment variables) and adjusts accordingly.

```bash
# Create virtual environment (local development only)
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
   git clone <repository-url>
   cd disaster-response-project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install local package** (required for training scripts):
   The project uses modern Python packaging with `pyproject.toml`:
   ```bash
   # Recommended - installs package in development mode
   pip install -e .
   # Or set PYTHONPATH per call (macOS/Linux)
   PYTHONPATH=src python scripts/04_create_production_model.py
   # PowerShell
   $env:PYTHONPATH = "src"; python scripts/04_create_production_model.py
   ```

3. **Download NLTK resources** (handled automatically):
   - punkt tokenizer
   - stopwords corpus
   - wordnet corpus

### Data Setup

1. **Process raw data**:
   ```bash
   python scripts/process_data.py data/01_raw/disaster_messages.csv data/01_raw/disaster_categories.csv data/02_stg/stg_disaster_response.db
   ```

2. **Train a model**:
   ```bash
   # Create production model
   python scripts/04_create_production_model.py
   
   # Alternative: Create experimental model with custom parameters
   python scripts/03_create_experimental_model.py
   ```

3. **Run the web application**:
   ```bash
   python run.py
   ```
   Open your browser to `http://localhost:5000`
   
   **Note**: Use `run.py` as the entry point (not `app/app.py`) as it properly handles the Flask application factory and configuration.

   Prerequisites for the app to start successfully:
   - Database present at `data/02_stg/stg_disaster_response.db` (run Data Setup if missing)
   - Model available at `model/disaster_rf_v1-2-0_prod_2025-09-11.pkl` or set `GDRIVE_MODEL_ID`

### Replit Deployment

The Flask application is optimized for production deployment on Replit with **Autoscale** and **Gunicorn** for maximum performance:

1. **Import the project** into your Replit workspace
2. **Dependencies automatically installed** during deployment build process
3. **Upload the model file** (if not using Google Drive):
   - **For files < 100MB**: Use Replit's GUI uploader (Files → Upload File)
   - **For files > 100MB**: Use `scp` (SSH) to upload directly:
     ```powershell
     # PowerShell (Windows)
     scp -i $env:USERPROFILE/.ssh/replit -P 22 model/your-model-file.pkl username@your-repl-id.replit.dev:~/
     ```
     ```bash
     # macOS/Linux
     scp -i ~/.ssh/replit -P 22 model/your-model-file.pkl username@your-repl-id.replit.dev:~/
     ```
     Then in Replit Shell, move it to the model directory:
     ```bash
     cd workspace
     mv ~/your-model-file.pkl model/
     ls -lh model/your-model-file.pkl  # Verify
     ```
   - **To find your Replit SSH connection info**: Run `echo $REPLIT_SSH_HOST` in Replit Shell
4. **Set environment variables** (optional if model is uploaded locally):
   - `GDRIVE_MODEL_ID`: Google Drive file ID for model download (only needed if model is not in `model/` directory)
   - Ensure the SQLite DB exists at `data/02_stg/stg_disaster_response.db` (upload it or adjust config)
5. **Run the application**: Click the "Run" button in Replit
6. **Access the app**: Use the provided Replit URL

**Note**: The app will use the local model file if present in `model/` directory. If not found and `GDRIVE_MODEL_ID` is set, it will download from Google Drive. The database file must exist for the app to function.

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

# Create experimental model
python scripts/03_create_experimental_model.py

# Compare experiment results
python scripts/compare_models.py

# Additional analysis tools
python scripts/evaluate_hierarchy.py
python scripts/optimize_hierarchy_threshold_reduction.py  # Hierarchy parameter optimization
python scripts/optimize_per_category_thresholds.py        # Per-category threshold optimization
python scripts/test_experimental_model.py
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
- **Hierarchy Demo**: Interactive demonstration of logical consistency enforcement
- **Data Visualization**: Interactive charts showing message distribution and categories
- **Model Performance**: Visual representation of model metrics
- **Responsive Design**: Tailwind CSS-based modern interface
- **Cloud Deployment**: Optimized for Replit deployment with automatic model downloading

### Hierarchy Processing Demo

The web application includes a live demonstration of the hierarchy post-processing system:

**Key Features:**
- **Toggle Interface**: Enable/disable hierarchy processing with a simple checkbox
- **Violation Detection**: Automatically identifies logical inconsistencies (e.g., `medical_help=YES` but `aid_related=NO`)
- **Before/After Comparison**: Visual diff table showing raw predictions vs. hierarchy-corrected results
- **Curated Examples**: Pre-filled messages that reliably trigger violations for demonstration
- **Production Metrics**: Static display of real performance data (zero violations on 26,027 test messages)

**Demo Flow:**
1. Enter a message (or use curated examples)
2. Submit without hierarchy processing to see raw AI predictions with violations
3. Enable hierarchy toggle and resubmit to see automatic corrections
4. Compare results in the side-by-side violation diff table

**Endpoints:**
- `/classify` - Main classification endpoint with optional `use_hierarchy` parameter
- Supports both form submissions and URL parameters for easy testing

This demonstrates the system's ability to enforce logical consistency in AI predictions for mission-critical disaster response scenarios.

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

**Note**: The `run.py` file serves as the application entry point and properly imports the Flask app from `app/app.py`.

#### Replit Deployment
The application is pre-configured for Replit deployment:
- **Automatic Model Download**: Downloads model from Google Drive if not present
- **Environment Variables**: Supports `GDRIVE_MODEL_ID` for model access
- **Port Configuration**: Automatically uses Replit's assigned port
- **Error Handling**: Robust error handling for cloud deployment scenarios

## 🚀 Production Deployment

The application is configured for optimal production performance using modern deployment practices:

### Deployment Architecture
- **Deployment Type**: Autoscale
  - Automatically scales based on traffic demand
  - Scales down to zero when idle (cost-effective)
  - Scales up automatically during high traffic
  - Pay-per-usage billing model

- **Production Server**: Gunicorn
  - **40x performance improvement** over Flask development server
  - **Multi-worker processing**: 2 workers handle concurrent requests
  - **Production-ready**: Optimized for real-world web traffic
  - **Timeout handling**: 120-second timeout for longer operations

### Technical Configuration
```bash
# Production command (automatically configured)
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 wsgi:application
```

### Files Added for Production
- **`wsgi.py`**: WSGI entry point for Gunicorn deployment
- **Deployment configuration**: Automated build and run commands

### Performance Benefits
| Metric | Development | Production | Improvement |
|--------|-------------|------------|-------------|
| Server Type | Flask dev server | Gunicorn | 40x faster |
| Concurrency | Single-threaded | Multi-worker | Concurrent requests |
| Scaling | Fixed resources | Auto-scale | Dynamic scaling |
| Cost Model | Always running | Pay-per-use | Cost-effective |

### When to Use Each Setup
- **Development**: Use `python run.py` for coding, debugging, local testing
- **Production**: Replit's deployment system automatically uses Gunicorn configuration
- **Local Production Testing**: Use Gunicorn command for local performance validation

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

### Hierarchy Post-Processing

The system includes a hierarchy post-processor that enforces parent-child consistency in multi-label predictions:

- **Parent ≥ Child Probabilities**: Ensures hierarchical relationships (e.g., `aid_related` ≥ `medical_help`)
- **Decision-Level Forcing**: If any child predicts positive, parent is forced positive
- **Critical Label Thresholds**: Reduced thresholds for safety-critical labels to improve recall
- **Violation Reduction**: Eliminates parent < child probability violations post-processing

API and Config
- API: `apply_hierarchy(probs, thresholds, taxonomy, critical_labels, exclude, critical_threshold_reduction=...)`
- Config default: `HIERARCHY_CRITICAL_THRESHOLD_REDUCTION = 0.0` (in `src/disasterproject/utils/config.py`). Scripts import and pass this value explicitly.
- Metrics: hierarchy violation rate is reported as "violations per 1k edges" (normalized by total parent→child edges evaluated), improving comparability across taxonomies.

Metric Definition Change
- As of 2025-09-18, "violations per 1k" is normalized by total parent→child edges evaluated (per-edge), not by samples.
- Earlier runs may show "per 1k samples". When comparing across runs, ensure you compare the same denominator.

Note on Edge Metrics
- Samples lacking complete probabilities (for any label) are excluded from hierarchy edge metrics to avoid mixing hard labels with probabilities. See the session note for details: `docs/sessions/active/2025-09-17-implement-hierarchy.md`.

Reproducibility: Persisted Thresholds
- During evaluation, the effective per-label thresholds used for hierarchy decisions are saved for reproducibility:
  - Production evaluation: `model/thresholds_used_hierarchy.json`
  - Experimental evaluator: `experiments/hierarchy_evaluation/thresholds_used_hierarchy_<timestamp>.json`
- The saved values reflect any configured critical-label reduction; with the current default (0.0), they typically remain 0.5 unless overridden by experiment thresholds.

#### Label Exclusions

**`child_alone` Label**: This category has 0 positive examples across all 26,027 messages in the dataset (0.000%). Due to this complete absence of training data, the `child_alone` label is excluded from hierarchy constraints to prevent spurious activations while remaining visible in model outputs for potential future use.

This design choice prioritizes model reliability by avoiding false positives in categories where the system has no learning signal, while maintaining the label structure for completeness.


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

## Dev hygiene

- pre-commit install
- pre-commit run --files <files>
- pytest -q

### Testing

See [docs/testing.md](docs/testing.md) for the full strategy, including marker usage and troubleshooting notes. Run the suites that match your change scope:

```bash
pytest -q -m "not gdrive and not perf and not slow"   # fast core
pytest -q -m perf                                    # performance suite
pytest -q -m gdrive                                  # Google Drive (mocked)
pytest -q                                            # entire suite
```

Marks (`gdrive`, `perf`, `slow`, etc.) keep the default CI lane fast while leaving opt-in coverage for deployment scenarios. Tests that require optional artifacts use descriptive skips so the signal stays clear even when resources are unavailable.

## 📁 Project Structure

```
disaster_response_project/
├── src/disasterproject/     # Core ML package
├── scripts/                     # Training and utility scripts
├── experiments/                 # Experiment results
├── app/                         # Web application
├── data/                        # Data storage
│   ├── 01_raw/                  # Raw input data
│   ├── 02_stg/                  # Staging/processed data
│   └── 04_fct/                  # Fact tables and final outputs
├── model/                       # Trained models and parameters
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
- **Tailwind CSS**: Frontend styling
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
python scripts/process_data.py data/01_raw/disaster_messages.csv data/01_raw/disaster_categories.csv data/02_stg/stg_disaster_response.db
```

**Port already in use:**
```bash
# macOS/Linux
export PORT=3001 && python run.py

# Windows (cmd)
set PORT=3001 && python run.py

# Windows (PowerShell)
$env:PORT=3001; python run.py
```

**Missing dependencies:**
```bash
# Reinstall requirements
pip install -r requirements.txt
```

**Package import errors (e.g., cannot import disasterproject):**
```bash
# Install local package
pip install -e .
# Or set PYTHONPATH
PYTHONPATH=src python scripts/04_create_production_model.py
```
