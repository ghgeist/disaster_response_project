# scripts/visualize_performance.py
# Standard library imports
from pathlib import Path

# Third-party imports
import pandas as pd
import plotly.graph_objs as go

def create_performance_comparison():
    """Generate interactive performance comparison charts."""
    fct_dir = Path("data/04_fct")
    
    # Load all experiment results
    experiments = {}
    for file in fct_dir.glob("fct_*_prediction_results.csv"):
        exp_name = file.stem.replace("fct_", "").replace("_prediction_results", "")
        df = pd.read_csv(file)
        # Focus on positive class metrics
        experiments[exp_name] = df[df['output_class'] == '1']
    
    # Create comparison visualizations
    # 1. Average metrics comparison
    # 2. Per-category recall improvements
    # 3. Precision-recall trade-off analysis
    
    return experiments

if __name__ == "__main__":
    create_performance_comparison()