"""Tests for the metrics IO utilities."""

from pathlib import Path

import pandas as pd

from disasterproject.utils.metrics_io import read_metrics_csv


def test_read_metrics_csv_success(tmp_path: Path) -> None:
    """It normalizes column names and coerces output_class to string."""
    df = pd.DataFrame(
        {
            "Output Class": [1],
            "Precision": [0.75],
            "Recall": [0.5],
        }
    )
    csv_path = tmp_path / "metrics.csv"
    df.to_csv(csv_path, index=False)

    result = read_metrics_csv(csv_path)

    assert result is not None
    assert list(result.columns) == ["output_class", "precision", "recall"]
    assert result.loc[0, "output_class"] == "1"


def test_read_metrics_csv_missing_file(tmp_path: Path) -> None:
    """Missing files resolve to ``None`` without raising."""
    missing = tmp_path / "absent.csv"

    assert read_metrics_csv(missing) is None


def test_read_metrics_csv_parse_error(tmp_path: Path) -> None:
    """Parse errors are logged and handled gracefully."""
    bad_csv = tmp_path / "broken.csv"
    bad_csv.write_text('col_a,col_b\n1,"unterminated')

    assert read_metrics_csv(bad_csv) is None
