import pandas as pd

def summarize_data(df):
    """
    Summarizes a DataFrame with various statistics.

    This function takes a DataFrame as input and calculates the following statistics:
    - Number of unique values per column
    - Total number of values per column
    - Number of missing values per column
    - Data completeness per column (percentage of non-missing values)
    - Percentage of unique values per column
    - Data type of each column

    The result is a new DataFrame with one row per original column and one column per statistic.

    Parameters:
    df (pd.DataFrame): The DataFrame to summarize.

    Returns:
    pd.DataFrame: A DataFrame summarizing the input.

    Raises:
    TypeError: If df is not a pandas DataFrame.
    ValueError: If df is empty.
    """
    # Check if df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input should be a pandas DataFrame")

    # Check if df is empty
    if df.empty:
        raise ValueError("Input DataFrame should not be empty")

    # Calculate unique_count and missing_count in one pass
    summary_df = df.agg(['nunique', 'count', lambda x: x.isnull().sum()]).transpose()

    # Rename the columns
    summary_df.columns = ['unique_values', 'total_values', 'values_missing']

    # Calculate the data completeness
    summary_df['data_completeness'] = round((summary_df['total_values'] / len(df)) * 100, 2)

    #Calculate the percentage of unique values and round to two decimal places
    summary_df['unique_percentage'] = round((summary_df['unique_values'] / summary_df['total_values']) * 100, 2)

    #Re-order the columns
    summary_df = summary_df[['total_values','unique_values', 'unique_percentage', 'values_missing', 'data_completeness']]

    # Add column types
    summary_df['type'] = df.dtypes

    # Reset the index
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={'index': 'columns'}, inplace=True)

    # Return the DataFrame
    return summary_df

def export_if_changed(df, filepath, dtypes=None):
    """
    Export a pandas DataFrame to a CSV file if it has changed compared to an existing file.

    Parameters:
    df (pandas.DataFrame): The DataFrame to export.
    filepath (str): The path to the existing file.
    dtypes (dict, optional): A dictionary specifying the data types of the columns in the DataFrame. Defaults to None.

    Returns:
    None

    Raises:
    FileNotFoundError: If the existing file is not found.

    """
    # Load the existing data
    try:
        existing_df = pd.read_csv(filepath, dtype=dtypes)
    except FileNotFoundError:
        existing_df = pd.DataFrame()

    # Check if the dataframes are different
    if not existing_df.equals(df):
        # If they are different, write the new data to the file
        df.to_csv(filepath, index=False)
        print(f"Data has changed. The file {filepath} has been updated.")
    else:
        print(f"No changes in the data. The file {filepath} has not been updated.")
