"""
Visualization helper functions for route handlers.
"""
import json
import logging
from typing import List, Tuple

import pandas as pd
import plotly

from app.services.metrics_service import load_metric_frames, extract_perf_triplet
from app.utils.formatting import format_request_context
from app.visualizations import ChartGenerator

logger = logging.getLogger(__name__)


def _create_basic_visualizations(data_service, chart_generator) -> Tuple[List[dict], List[str]]:
    """
    Create basic genre and message type visualizations.

    Args:
        data_service: DataService instance.
        chart_generator: ChartGenerator instance.

    Returns:
        Tuple of (graphs_list, descriptions_list).
    """
    df = data_service.get_data()

    genre_names, genre_related_counts = chart_generator.prepare_genre_data(df)
    genre_graph = chart_generator.create_genre_visual(genre_names, genre_related_counts)

    message_types_df = chart_generator.classify_message_types(df)
    message_type_graph = chart_generator.plot_message_types(message_types_df)

    graphs = [genre_graph, message_type_graph]
    descriptions = [
        (
            "Direct messages dominate disaster communications. Bars show counts by source, "
            "stacked by disaster-related vs not. The predominance of direct messages "
            "underscores the need to triage individual cries for help."
        ),
        (
            "Among disaster-related direct messages, requests for aid are far more common "
            "than offers; direct reports are frequent. The model must reliably identify "
            "these requests."
        ),
    ]

    return graphs, descriptions


def _add_performance_visualization(
    graphs: List[dict],
    descriptions: List[str],
) -> Tuple[List[dict], List[str]]:
    """
    Add performance visualization if data is available.

    Args:
        graphs: List of existing graphs.
        descriptions: List of existing descriptions.

    Returns:
        Tuple of (updated_graphs, updated_descriptions).
    """
    context = format_request_context()
    try:
        base_df, opt_df = load_metric_frames()

        if base_df is not None and opt_df is not None:
            metrics, labels = extract_perf_triplet(base_df, opt_df)
            perf_graph = ChartGenerator.create_performance_visual(metrics, labels)
            graphs.append(perf_graph)
            descriptions.append(
                (
                    "Baseline (blue) vs Optimized (orange). Precision improves slightly; "
                    "recall drops significantly. In disasters, missing real help messages "
                    "is costly."
                )
            )
        else:
            logger.warning("Performance CSVs missing; skipping performance chart%s", context)
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as perf_exc:
        logger.warning(
            "Skipping performance chart due to data issue%s: %s",
            context,
            perf_exc,
        )
    except Exception as perf_exc:
        logger.warning(
            "Skipping performance chart due to unexpected error%s: %s",
            context,
            perf_exc,
        )

    return graphs, descriptions


def _encode_graphs_to_json(graphs: List[dict]) -> Tuple[str, List[str]]:
    """
    Encode plotly graphs to JSON format.

    Args:
        graphs: List of plotly graph objects.

    Returns:
        Tuple of (graph_json_string, ids_list).
    """
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    try:
        graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
        return graph_json, ids
    except (TypeError, ValueError) as json_error:
        logger.error("Error encoding graphs to JSON: %s", json_error)
        return "[]", []
