# Standard library imports
import json
import os
import sys

# Third-party imports
from flask import Flask, render_template, request, jsonify
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
import joblib
import pandas as pd
import plotly
from plotly.graph_objs import Bar

# Local application imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.train_classifier import tokenize
from app.graph_generator import *

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data\\02_stg\\stg_disaster_response.db')
df = pd.read_sql_table('stg_disaster_response', engine)

# load model
model = joblib.load("..\\models\\classifier.pkl")


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
    #create genre graph
    genre_names, genre_related_counts = prepare_genre_data(df)
    genre_graph = create_genre_visual(genre_names, genre_related_counts)

    #Create message_type graph
    message_types_df = classify_message_types(df)
    message_type_graph = plot_message_types(message_types_df)

    #Create list of visuals
    graphs = [genre_graph, message_type_graph]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

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
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

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