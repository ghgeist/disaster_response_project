import warnings
import pandas as pd
import plotly.graph_objs as go

def prepare_genre_data(df):
    """
    Prepare genre data for visualization.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'genre', 'related', and 'message' columns.

    Returns:
    genre_names (list): List of genre names sorted by count.
    genre_related_counts (pd.DataFrame): DataFrame of counts of 'related' per genre.
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

    return genre_names, genre_related_counts

def create_genre_visual(genre_names, genre_related_counts):
    """
    Create a Plotly graph object for genre data.

    Parameters:
    genre_names (list): List of genre names.
    genre_related_counts (pd.DataFrame): DataFrame of counts of 'related' per genre.

    Returns:
    genre_graph (dict): Dictionary representing a Plotly graph object.
    """
    # Create a dictionary to map 'related' values to new names
    related_names = {0: 'not related', 1: 'related', 2: 'ambiguous'}

    # Create visuals
    genre_graph = {
        'data': [
            go.Bar(
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
    return genre_graph

def classify_message_types(df):
    """
    Classify message types based on 'request', 'offer', and 'direct_report' columns.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'message', 'request', 'offer', 'direct_report', and other columns.

    Returns:
    df_filtered (pd.DataFrame): Filtered DataFrame with a new 'message_type' column and duplicates dropped.
    """
    # Filter df where genre is 'direct', related is 1 and request is 1
    df_filtered = df[(df['genre'] == 'direct') & (df['related'] == 1)]

    # Drop all columns except 'message', 'request', 'offer', 'direct_report'
    df_filtered = df_filtered[['message', 'request', 'offer', 'direct_report']]

    # Create a new column called 'message_type' which returns 'request' if request is 1, 'offer' if offer is 1 and 'direct_report' if direct_report is 1
    df_filtered['message_type'] = df_filtered.apply(
        lambda x: 'offer' if x['offer'] == 1 else ('request' if x['request'] == 1 else ('direct_report' if x['direct_report'] == 1 else 'other')),
        axis=1
    )

    # Drop the 'request', 'offer' and 'direct_report' columns
    df_filtered = df_filtered.drop(columns=['request', 'offer', 'direct_report'])

    # Check if the number of duplicates is greater than 1% of the DataFrame's length
    num_duplicates = df_filtered.duplicated().sum()
    if num_duplicates > len(df_filtered) * 0.01:
        warnings.warn(f"Dropping {num_duplicates} duplicates, which is more than 1% of the DataFrame's length")

    # Drop duplicates
    df_filtered = df_filtered.drop_duplicates()

    return df_filtered

def plot_message_types(df):
    """
    Create a Plotly graph object for message type data.

    Parameters:
    df (pd.DataFrame): DataFrame containing a 'message_type' column.

    Returns:
    graph (dict): Dictionary representing a Plotly graph object.
    """
    # Count the number of messages per message_type
    message_types_count = df['message_type'].value_counts().sort_values(ascending=True)

    # Create a dictionary representing a Plotly graph object
    graph = {
        'data': [
            go.Bar(
                y=message_types_count.index.tolist(),
                x=message_types_count.values.tolist(),
                name='Count',
                orientation='h'
            )
        ],

        'layout': {
            'title': 'Number of Messages per Message Type',
            'yaxis': {
                'title': "Message Type",
                'automargin': True  # This line ensures that the y-axis labels fit into the layout
            },
            'xaxis': {
                'title': "Number of Messages"
            },
            'barmode': 'stack',
            'autosize': False,
            'width': 1000,  # Adjust the width of the graph
            'height': 600,  # Adjust the height of the graph
            'margin': {
                'l': 100,  # Increase left margin to make more room for y-axis labels
                'r': 50,
                'b': 100,
                't': 100,
                'pad': 4
            },
            }
    }

    return graph
