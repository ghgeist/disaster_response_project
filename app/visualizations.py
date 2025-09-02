"""
Chart generation utilities for the disaster response application.
"""
import warnings
import pandas as pd
import plotly.graph_objs as go
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Service for generating visualization charts."""
    
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
            colors = {'Not Related': '#A0AEC0', 'Related': '#3182CE', 'Ambiguous': '#F59E0B'}

            # Create visuals
            genre_graph = {
                'data': [
                    go.Bar(
                        y=genre_names,
                        x=genre_related_counts[col],
                        name=related_names[col],
                        orientation='h',
                        marker=dict(color=colors[related_names[col]])
                    )
                    for col in genre_related_counts.columns
                ],

                'layout': {
                    'title': {
                        'text': 'Message Genre Distribution',
                        'font': {'color': '#E2E8F0'}
                    },
                    'xaxis': {
                        'title': 'Message Count',
                        'gridcolor': '#2D3748',
                        'color': '#A0AEC0'
                    },
                    'yaxis': {
                        'title': 'Genre',
                        'color': '#A0AEC0'
                    },
                    'barmode': 'stack',
                    'paper_bgcolor': '#2D3748',
                    'plot_bgcolor': '#2D3748',
                    'legend': {
                        'font': {'color': '#E2E8F0'}
                    },
                    'margin': {'l': 80, 'r': 50, 't': 80, 'b': 50}
                }
            }
            return genre_graph
            
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
            graph = {
                'data': [
                    go.Bar(
                        y=message_types_count.index.tolist(),
                        x=message_types_count.values.tolist(),
                        name='Count',
                        orientation='h',
                        marker=dict(color='#3182CE')
                    )
                ],

                'layout': {
                    'title': {
                        'text': 'Direct Message Types',
                        'font': {'color': '#E2E8F0'}
                    },
                    'yaxis': {
                        'title': "Message Type",
                        'automargin': True,
                        'color': '#A0AEC0'
                    },
                    'xaxis': {
                        'title': "Number of Messages",
                        'gridcolor': '#2D3748',
                        'color': '#A0AEC0'
                    },
                    'barmode': 'stack',
                    'paper_bgcolor': '#2D3748',
                    'plot_bgcolor': '#2D3748',
                    'margin': {'l': 100, 'r': 50, 't': 80, 'b': 50, 'pad': 4}
                }
            }

            return graph
            
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

            trace_base = go.Bar(x=categories, y=base_vals, name=labels[0])
            trace_opt = go.Bar(x=categories, y=opt_vals, name=labels[1])

            layout = go.Layout(
                title={"text": "Baseline vs Optimized Model Performance", "font": {"color": "#E2E8F0"}},
                yaxis=dict(title="Score (%)", rangemode="tozero", gridcolor="#2D3748", color="#A0AEC0"),
                xaxis=dict(color="#A0AEC0"),
                barmode="group",
                paper_bgcolor="#2D3748",
                plot_bgcolor="#2D3748",
                font=dict(color="#E2E8F0"),
                legend=dict(orientation="h", x=0.5, xanchor="center"),
                margin={"l": 60, "r": 40, "t": 60, "b": 60},
            )
            return {"data": [trace_base, trace_opt], "layout": layout}
        except Exception as e:
            logger.error(f"Error creating performance visual: {e}")
            raise

    @staticmethod
    def apply_dark_layout(fig_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure existing figure dict adopts dark theme styles."""
        try:
            layout = fig_dict.get("layout", {})
            layout.update({
                "paper_bgcolor": "#2D3748",
                "plot_bgcolor": "#2D3748",
                "font": {"color": "#E2E8F0"},
            })
            # Make axes readable
            if "xaxis" in layout:
                xax = layout["xaxis"]
                if isinstance(xax, dict):
                    xax.setdefault("color", "#A0AEC0")
                    xax.setdefault("gridcolor", "#2D3748")
            if "yaxis" in layout:
                yax = layout["yaxis"]
                if isinstance(yax, dict):
                    yax.setdefault("color", "#A0AEC0")
                    yax.setdefault("gridcolor", "#2D3748")
            fig_dict["layout"] = layout
            return fig_dict
        except Exception as e:
            logger.error(f"Error applying dark layout: {e}")
            raise