# Standard library imports
import os
import sys
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
from models.train_classifier import tokenize

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
    This function is linked to the home page of the web application. It renders the 'master.html' template and 
    passes the necessary data to generate Plotly graphs on the page.

    The function first extracts data for visuals from the global dataframe 'df'. It then creates a list of Plotly 
    graphs, which are encoded in JSON format using Plotly's JSON encoder. The JSON graphs and their corresponding 
    ids are passed to the 'master.html' template for rendering.

    Returns:
        A rendered HTML template ('master.html') with data for visuals.
    """
    # Group by 'genre' and 'related', count 'message', unstack 'related', and sum across rows
    genre_counts = df.groupby(['genre', 'related']).count()['message'].unstack().sum(axis=1)

    # Sort genres by count
    sorted_genres = genre_counts.sort_values(ascending=True).index

    # Get sorted genre names
    genre_names = list(sorted_genres)

    # Group by 'genre' and 'related' again, count 'message', and unstack 'related'
    genre_related_counts = df.groupby(['genre', 'related']).count()['message'].unstack()

    # Reindex with sorted genre names
    genre_related_counts = genre_related_counts.reindex(genre_names)
    
    # Create a dictionary to map 'related' values to new names
    related_names = {0: 'not related', 1: 'related', 2: 'ambiguous'}    

    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    y=genre_names,
                    x=genre_related_counts[col],
                    name=related_names[col],  # Use mapped name
                    orientation='h'
                )
                for col in genre_related_counts.columns
            ],

            'layout': {
                'title': 'Messages per Genre and Relatedness',
                'xaxis': {
                    'title': "Count"
                },
                'yaxis': {
                    'title': "Genre"
                },
                'barmode': 'stack'
            }
        }
    ]
    
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