# Standard library imports
import os
import sys
# Add the parent directory to sys.path to allow importing from there.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json

# Third-party imports
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import plotly
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Local application imports
# Import the tokenize function from the local train_classifier module.
from models.train_classifier import tokenize

# Create a Flask web server from the current file.
app = Flask(__name__)

# Load data from the SQLite database.
engine = create_engine('sqlite:///../data\\02_stg\\stg_disaster_response.db')
df = pd.read_sql_table('stg_disaster_response', engine)

# Load the trained machine learning model.
model = joblib.load("..\\models\\classifier.pkl")

# Define the main page route.
@app.route('/')
@app.route('/index')
def index():
    """
    This function is linked to the home page of the web application. It renders the 'master.html' template and 
    passes the necessary data to generate Plotly graphs on the page.

    The function first extracts data for visuals from the global dataframe 'df'. It then creates a list of Plotly 
    graphs, which are encoded in JSON format using Plotly's JSON encoder. The JSON graphs and their corresponding 
    ids are passed to the 'master.html' template for rendering.

    Returns:
        A rendered HTML template ('master.html') with data for visuals.
    """
    # Extract data needed for visuals.
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Create visuals for the web page.
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # Encode plotly graphs in JSON.
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render the web page with plotly graphs.
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Define the page route that handles user query and displays model results.
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
    # Save user input in query.
    query = request.args.get('query', '') 

    # Use model to predict classification for query.
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the go.html template and pass the data into the template.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

# Define the main function that starts the Flask web server.
def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

# If this file is executed directly, start the web server.
if __name__ == '__main__':
    main()