# Standard library imports
import json
import os
import sys

# Third-party imports
from flask import Flask, render_template, request, jsonify, abort, send_from_directory
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
import joblib
import pandas as pd
import plotly
from plotly.graph_objs import Bar
import requests

# Local application imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.train_classifier import tokenize
from app.graph_generator import *

app = Flask(__name__)

# Serve favicon: prefer .ico, gracefully fall back to .png variants
@app.route('/favicon.ico')
def favicon():
    app_dir = os.path.dirname(__file__)
    images_dir = os.path.abspath(os.path.join(app_dir, '..', 'images'))
    ico_path = os.path.join(images_dir, 'favicon.ico')
    png_fallbacks = ['favicon.png', 'image.png']

    if os.path.exists(ico_path):
        return send_from_directory(images_dir, 'favicon.ico', mimetype='image/x-icon')

    for png_name in png_fallbacks:
        png_path = os.path.join(images_dir, png_name)
        if os.path.exists(png_path):
            return send_from_directory(images_dir, png_name, mimetype='image/png')

    abort(404)

#TO DO: Change this so you can specific the model that you want the app to run
# load data
try:
    app_dir = os.path.dirname(__file__)
    db_path = os.path.abspath(os.path.join(app_dir, '..', 'data', '02_stg', 'stg_disaster_response.db'))
    engine = create_engine(f'sqlite:///{db_path}')
    df = pd.read_sql_table('stg_disaster_response', engine)
except Exception as e:
    print(f"Error loading data from database: {e}", file=sys.stderr)
    sys.exit(1)

def download_model_if_missing() -> str:
    """Ensure the classifier.pkl exists locally; download from Drive if missing."""
    app_dir = os.path.dirname(__file__)
    model_path = os.path.abspath(os.path.join(app_dir, '..', 'models', 'classifier.pkl'))
    
    if os.path.exists(model_path):
        return model_path
    
    print("Model not found locally, downloading from Google Drive...", file=sys.stderr)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    file_id = os.environ.get('GDRIVE_MODEL_ID')
    if not file_id or file_id.strip() in {'', 'YOUR_FILE_ID', 'YOUR_GOOGLE_DRIVE_FILE_ID'}:
        raise RuntimeError(
            "GDRIVE_MODEL_ID is not set or is using a placeholder. "
            f"Provide a valid Google Drive file ID via the GDRIVE_MODEL_ID env var, "
            f"or place the model at: {model_path}"
        )
    
    # Create temporary file for download
    temp_path = f"{model_path}.tmp"
    
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            
            # Check if response is HTML (Google Drive warning page)
            content_type = r.headers.get('content-type', '')
            if 'text/html' in content_type.lower():
                raise RuntimeError(
                    "Google Drive returned HTML instead of the model file. "
                    "This usually means the file requires authentication or is too large. "
                    "Please check the GDRIVE_MODEL_ID or download manually."
                )
            
            # Download to temporary file first
            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        
        # Validate the downloaded file
        if os.path.getsize(temp_path) < 1000:  # Model files should be at least 1KB
            raise RuntimeError("Downloaded file is too small, likely corrupted")
        
        # Try to load the model to validate it's not corrupted
        try:
            test_model = joblib.load(temp_path)
            del test_model  # Clean up test load
        except Exception as e:
            raise RuntimeError(f"Downloaded model file is corrupted: {e}")
        
        # If validation passes, move temp file to final location
        os.replace(temp_path, model_path)
        print("Model downloaded and validated successfully!", file=sys.stderr)
        
    except Exception as e:
        # Clean up temporary file on any error
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass  # Ignore cleanup errors
        
        # Provide helpful error message
        if "timeout" in str(e).lower():
            raise RuntimeError(
                f"Download timed out. Please check your internet connection and try again. "
                f"Error: {e}"
            )
        elif "corrupted" in str(e).lower():
            raise RuntimeError(
                f"Download failed due to corruption. Please try again. "
                f"Error: {e}"
            )
        else:
            raise RuntimeError(
                f"Failed to download model: {e}. "
                f"Please check the GDRIVE_MODEL_ID or download manually to: {model_path}"
            )
    
    return model_path

# load model
try:
    model_path = download_model_if_missing()
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    sys.exit(1)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    This function handles the main page of the web application. It prepares the data for the genre and message type
    graphs, encodes the graphs in JSON format, and renders the 'master.html' template with the graph data.

    The function first prepares the data for the genre and message type graphs using the 'prepare_genre_data' and
    'classify_message_types' functions respectively. It then creates the graphs using the 'create_genre_visual' and
    'plot_message_types' functions.

    The graphs are then encoded in JSON format using the 'json.dumps' function with 'plotly.utils.PlotlyJSONEncoder'
    as the encoder class.

    Finally, the function renders the 'master.html' template, passing the graph IDs and the JSON-encoded graph data
    to the template.

    Returns:
        A rendered HTML template ('master.html') with the graph IDs and JSON-encoded graph data.
    """
    try:
        # create genre graph
        genre_names, genre_related_counts = prepare_genre_data(df)
        genre_graph = create_genre_visual(genre_names, genre_related_counts)

        # Create message_type graph
        message_types_df = classify_message_types(df)
        message_type_graph = plot_message_types(message_types_df)

        # Create list of visuals
        graphs = [genre_graph, message_type_graph]

        # encode plotly graphs in JSON
        ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
        graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    except Exception as err:
        abort(500, description=f"Error preparing data for visualization: {err}")

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    This function handles the user's query and uses the trained model to classify the query.

    The function first retrieves the user's query from the request arguments. It then uses the global 'model'
    to predict the classification labels for the query. The labels are matched with the corresponding columns
    from the global 'df' dataframe to create a dictionary of classification results.

    The function then renders the 'go.html' template, passing the user's query and the classification results
    to the template.

    Returns:
        A rendered HTML template ('go.html') with the user's query and classification results.
    """
    try:
        # save user input in query
        query = request.args.get('query', '')

        # use model to predict classification for query
        classification_labels = model.predict([query])[0]
        classification_results = dict(zip(df.columns[4:], classification_labels))

    except Exception as e:
        return render_template('error.html', message=f"Error processing query: {e}")

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    """
    Runs the application on the specified host and port.

    Parameters:
    - host (str): The host IP address to run the application on.
    - port (int): The port number to run the application on.
    - debug (bool): Whether to enable debug mode or not.

    Returns:
    None
    """
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()