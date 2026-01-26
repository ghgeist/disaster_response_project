"""
Data service for loading and caching database content.
"""
from __future__ import annotations

import logging

import pandas as pd
import sqlalchemy.exc
from sqlalchemy import create_engine

from disasterproject.utils.config import TARGET_COLUMNS

from .errors import DataServiceError

logger = logging.getLogger(__name__)


class DataService:
    """Service for managing data loading and operations."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._engine = None
        self._df = None

    @property
    def engine(self):
        """Get database engine, creating if necessary."""
        if self._engine is None:
            self._engine = create_engine(self.database_url)
        return self._engine

    def load_data(self, table_name: str = "stg_disaster_response") -> pd.DataFrame:
        """Load data from the database."""
        if self._df is not None:
            return self._df

        try:
            self._df = pd.read_sql_table(table_name, self.engine)
            logger.info("Data loaded successfully from table '%s'", table_name)
            return self._df

        except (OSError, pd.errors.DatabaseError, sqlalchemy.exc.SQLAlchemyError) as error:
            logger.error("Error loading data from database: %s", error)
            raise DataServiceError("Failed to load data from database.") from error

    def get_data(self) -> pd.DataFrame:
        """Get the loaded data."""
        if self._df is None:
            self.load_data()
        return self._df

    def get_category_columns(self) -> list:
        """Get the category column names using TARGET_COLUMNS for consistency."""
        df = self.get_data()
        available_columns = set(df.columns)
        return [col for col in TARGET_COLUMNS if col in available_columns]
