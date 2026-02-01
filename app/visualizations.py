"""
Chart generation utilities for the disaster response application.
"""
import logging
import warnings
from typing import Any, Dict, Tuple

import pandas as pd
import plotly.graph_objs as go

logger = logging.getLogger(__name__)


def _create_base_layout(title: str, xaxis_title: str, yaxis_title: str) -> go.Layout:
    """Creates a base layout for Plotly charts with a dark theme."""
    return go.Layout(
        title={'text': title, 'font': {'color': '#E2E8F0'}},
        xaxis={'title': xaxis_title, 'gridcolor': '#2D3748', 'color': '#A0AEC0'},
        yaxis={'title': yaxis_title, 'gridcolor': '#2D3748', 'color': '#A0AEC0'},
        paper_bgcolor='#1A202C',
        plot_bgcolor='#1A202C',
        legend={'font': {'color': '#E2E8F0'}},
        margin={'l': 80, 'r': 50, 't': 80, 'b': 50}
    )

def _create_bar_trace(x_data, y_data, name, orientation='h', color=None) -> go.Bar:
    """Creates a bar trace for Plotly charts."""
    marker = dict(color=color) if color else {}
    return go.Bar(
        x=x_data,
        y=y_data,
        name=name,
        orientation=orientation,
        marker=marker
    )


class ChartGenerator:
    """Service for generating visualization charts."""

    COLOR_PALETTE = {
        'primary': '#06d6a0',
        'secondary': '#118ab2',
        'tertiary': '#ffd166',
        'quaternary': '#ef476f',
        'neutral': '#A0AEC0'
    }

    @staticmethod
    def prepare_genre_data(df: pd.DataFrame) -> Tuple[list, pd.DataFrame]:
        """
        Prepare genre data for visualization.

        Args:
            df: DataFrame containing 'genre', 'related', and 'message' columns.

        Returns:
            Tuple of (genre_names, genre_related_counts)
        """
        try:
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

        except Exception as e:
            logger.error(f"Error preparing genre data: {e}")
            raise

    @staticmethod
    def create_genre_visual(genre_names: list, genre_related_counts: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a Plotly graph object for genre data.

        Args:
            genre_names: List of genre names.
            genre_related_counts: DataFrame of counts of 'related' per genre.

        Returns:
            Dictionary representing a Plotly graph object.
        """
        try:
            # Create a dictionary to map 'related' values to new names
            related_names = {0: 'Not Related', 1: 'Related', 2: 'Ambiguous'}
            colors = {
                'Related': ChartGenerator.COLOR_PALETTE['primary'],
                'Not Related': ChartGenerator.COLOR_PALETTE['secondary'],
                'Ambiguous': ChartGenerator.COLOR_PALETTE['tertiary']
            }

            # Create visuals
            traces = [
                _create_bar_trace(
                    x_data=genre_related_counts[col],
                    y_data=genre_names,
                    name=related_names[col],
                    color=colors[related_names[col]]
                )
                for col in genre_related_counts.columns
            ]

            layout = _create_base_layout(
                'Message Genre Distribution',
                'Message Count',
                'Genre'
            )
            layout.update(barmode='stack')
            layout.update(legend=dict(orientation='h', yanchor='top', y=-0.2, xanchor='center', x=0.5))

            return {'data': traces, 'layout': layout}

        except Exception as e:
            logger.error(f"Error creating genre visual: {e}")
            raise

    @staticmethod
    def classify_message_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify message types based on 'request', 'offer', and 'direct_report' columns.

        Args:
            df: DataFrame containing 'message', 'request', 'offer', 'direct_report', and other columns.

        Returns:
            Filtered DataFrame with a new 'message_type' column and duplicates dropped.
        """
        try:
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

        except Exception as e:
            logger.error(f"Error classifying message types: {e}")
            raise

    @staticmethod
    def plot_message_types(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a Plotly graph object for message type data.

        Args:
            df: DataFrame containing a 'message_type' column.

        Returns:
            Dictionary representing a Plotly graph object.
        """
        try:
            # Count the number of messages per message_type
            message_types_count = df['message_type'].value_counts().sort_values(ascending=True)

            # Create a dictionary representing a Plotly graph object
            trace = _create_bar_trace(
                x_data=message_types_count.values.tolist(),
                y_data=message_types_count.index.tolist(),
                name='Count',
                color=ChartGenerator.COLOR_PALETTE['primary']
            )

            layout = _create_base_layout(
                'Direct Message Types',
                'Number of Messages',
                'Message Type'
            )
            layout.yaxis.update(automargin=True)
            layout.margin.update(l=100, pad=4)


            return {'data': [trace], 'layout': layout}

        except Exception as e:
            logger.error(f"Error plotting message types: {e}")
            raise

    # --- Performance Deep Dive visualizations ---
    @staticmethod
    def create_performance_visual(metrics_dict: Dict[str, list], labels: list) -> Dict[str, Any]:
        """Create grouped bar chart comparing baseline vs optimized Precision/Recall/F1."""
        try:
            categories = ["Precision", "Recall", "F1"]
            base_vals = [metrics_dict.get("precision", [0, 0])[0], metrics_dict.get("recall", [0, 0])[0], metrics_dict.get("f1", [0, 0])[0]]
            opt_vals = [metrics_dict.get("precision", [0, 0])[1], metrics_dict.get("recall", [0, 0])[1], metrics_dict.get("f1", [0, 0])[1]]

            trace_base = _create_bar_trace(categories, base_vals, labels[0], orientation='v', color=ChartGenerator.COLOR_PALETTE['secondary'])
            trace_opt = _create_bar_trace(categories, opt_vals, labels[1], orientation='v', color=ChartGenerator.COLOR_PALETTE['primary'])

            layout = _create_base_layout(
                'Baseline vs Optimized Model Performance',
                '',
                'Score (%)'
            )
            layout.update(barmode='group', legend=dict(orientation='h', x=0.5, xanchor='center'))
            layout.yaxis.update(rangemode='tozero')
            layout.margin.update(l=60, r=40, t=60, b=60)

            return {"data": [trace_base, trace_opt], "layout": layout}
        except Exception as e:
            logger.error(f"Error creating performance visual: {e}")
            raise
