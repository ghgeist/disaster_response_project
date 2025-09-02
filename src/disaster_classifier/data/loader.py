"""
Data loading functions for disaster response classification.
"""

import logging
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from ..utils.config import TARGET_COLUMNS


def load_data(db_filepath):
    """
    Load data from a SQLite database.

    This function reads a table from a SQLite database and splits it into features (X) and labels (y). 
    The features are the 'message' column of the table, and the labels are the columns specified by TARGET_COLUMNS.
    If any of the TARGET_COLUMNS contain NaN values, a ValueError is raised.

    Args:
    db_filepath (str): The file path of the SQLite database.

    Returns:
    X (numpy.ndarray): The features from the 'message' column of the table.
    y (numpy.ndarray): The labels from the columns specified by TARGET_COLUMNS.

    Raises:
    ValueError: If any of the TARGET_COLUMNS contain NaN values.
    """
    try:
        database_url = "sqlite:///" + db_filepath.replace("\\", "/")
        engine = create_engine(database_url)
    except OperationalError:
        logging.error("Error connecting to database at %s", db_filepath)
        return None, None

    table_name = os.path.splitext(os.path.basename(db_filepath))[0]

    try:
        df = pd.read_sql_table(table_name, engine)
    except ValueError:
        logging.error("Table %s not found in database", table_name)
        return None, None

    try:
        X = df.message.values
        y = df[TARGET_COLUMNS].values

        nan_columns = df[TARGET_COLUMNS].isna().any()
        nan_columns_list = nan_columns[nan_columns == True].index.tolist()

        if len(nan_columns_list) > 0:
            logging.error("Columns with NaN values: %s", nan_columns_list)
            raise ValueError(
                "NaN values found in columns: %s. Check the TARGET_COLUMNS to make sure they are set up correctly "
                "or the underlying data" % nan_columns_list
            )

    except KeyError as e:
        logging.error("Column %s not found in table", e.args[0])
        return None, None
    except ValueError as e:
        logging.error(e)
        return None, None

    return X, y
