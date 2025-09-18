"""Utility helpers for loading persisted model metrics."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def read_metrics_csv(path: Path) -> Optional[pd.DataFrame]:
    """Return a normalized metrics DataFrame or ``None`` when the file is unavailable."""
    try:
        if not path.exists():
            logger.warning("Metrics CSV not found: %s", path)
            return None

        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        if "output_class" in df.columns:
            df["output_class"] = df["output_class"].astype(str)
        return df

    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        logger.error("File not found or empty metrics CSV %s: %s", path, exc)
        return None
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        logger.error("Parse error in metrics CSV %s: %s", path, exc)
        return None
    except Exception:
        logger.exception("Unexpected error reading metrics CSV %s", path)
        return None
