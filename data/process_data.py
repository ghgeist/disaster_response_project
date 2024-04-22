#Standard Imports
import logging
import os
import sys

#Third-party Imports
import pandas as pd
from sqlalchemy import create_engine

# Set up logging
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

fileHandler = logging.FileHandler('app.log')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

def load_data(messages_filepath, categories_filepath):
    """
    Load data from messages and categories csv files.

    Parameters:
    messages_filepath (str): The path to the messages csv file.
    categories_filepath (str): The path to the categories csv file.

    Returns:
    tuple: A tuple containing two dataframes, one for messages and one for categories. 

    Raises:
    FileNotFoundError: If either of the files does not exist.
    Exception: If any other error occurs during loading.
    """
    try:
        # Load messages dataset
        messages_df = pd.read_csv(messages_filepath)
        # Load categories dataset
        categories_df = pd.read_csv(categories_filepath)
    except FileNotFoundError as fnf_error:
        logging.error("File not found: %s", fnf_error)
        raise
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise
    return messages_df, categories_df

def process_categories(categories_df):
    """
    Process the categories dataframe by splitting the categories and converting values to 0 or 1.

    Parameters:
    categories_df (DataFrame): The categories dataframe to process.

    Returns:
    DataFrame: The processed categories dataframe. If an error occurs during processing, returns None.

    Raises:
    Exception: If an error occurs during processing.
    """
    try:
        # Set 'id' as the index
        categories_df.set_index('id', inplace=True)

        # Split the categories
        split_categories = categories_df['categories'].str.split(';', expand=True)

        # Use the first row to create column names
        row = split_categories.iloc[0]
        category_colnames = row.apply(lambda x: x[:-2])
        split_categories.columns = category_colnames

        # Convert category values to just numbers 0 or 1
        for column in split_categories:
            split_categories[column] = split_categories[column].str[-1]
            split_categories[column] = split_categories[column].astype(int)

    except Exception as e:
        logging.error("Error processing categories: %s", e)
        raise

    return split_categories

def process_messages(messages_df):
    """
    Process the messages dataframe by setting 'id' as the index.

    Parameters:
    messages_df (DataFrame): The messages dataframe to process.

    Returns:
    DataFrame: The processed messages dataframe. If an error occurs during processing, returns None.

    Raises:
    Exception: If an error occurs during processing.
    """
    try:
        # Set 'id' as the index
        messages_df.set_index('id', inplace=True)
    except Exception as e:
        logging.error("Error processing messages: %s", e)
        raise

    return messages_df

def join_messages_and_categories(messages_df, categories_df):
    """
    Join the messages and categories dataframes on their indices.

    Parameters:
    messages_df (DataFrame): The messages dataframe.
    categories_df (DataFrame): The categories dataframe.

    Returns:
    DataFrame: The dataframe resulting from joining messages and categories. 
               If an error occurs during the merge, returns None.

    Raises:
    Exception: If an error occurs during the merge.
    """
    try:
        categorized_messages_df = messages_df.merge(categories_df, left_index=True, right_index=True)
    except Exception as e:
        logging.error("Error joining messages and categories: %s", e)
        raise

    return categorized_messages_df

def drop_ambigious_messages(df):
    """
    Drop rows from the dataframe where the 'related' column is 2, indicating an ambiguous message.

    Parameters:
    df (DataFrame): The dataframe from which to drop rows.

    Returns:
    DataFrame: The dataframe with ambiguous messages dropped. If an error occurs during the process, returns None.

    Raises:
    Exception: If an error occurs during the process.
    """
    try:
        num_related_rows_dropped = df[df['related'] == 2].shape[0]
        percent_related_rows_dropped = round((num_related_rows_dropped / df.shape[0]) * 100,2)
        logging.info(
            "Dropping %s rows where 'related' is 2 which indicates that the message is ambiguous. "
            "This is %s%% of the data.",
            num_related_rows_dropped, percent_related_rows_dropped)
        df = df[df['related'] != 2]
    except Exception as e:
        logging.error("Error dropping ambiguous messages: %s", e)
        raise

    return df

def drop_duplicates(df):
    """
    Drop duplicate rows from the dataframe.

    Parameters:
    df (DataFrame): The dataframe from which to drop duplicates.

    Returns:
    DataFrame: The dataframe with duplicates dropped. If an error occurs during the process, returns None.

    Raises:
    Exception: If an error occurs during the process.
    """
    try:
        num_duplicate_rows = df.duplicated().sum()
        percent_duplicate_rows = round((num_duplicate_rows / df.shape[0]) * 100,2)
        logging.info("Dropping %s duplicate rows (%s%% of the data)", num_duplicate_rows, percent_duplicate_rows)
        df = df.drop_duplicates()
    except Exception as e:
        logging.error("Error dropping duplicates: %s", e)
        raise

    return df

def clean_data(messages_df, categories_df):
    """
    Clean the messages and categories dataframes by processing them, joining them, 
    dropping ambiguous messages, and dropping duplicates.

    Parameters:
    messages_df (DataFrame): The messages dataframe to clean.
    categories_df (DataFrame): The categories dataframe to clean.

    Returns:
    DataFrame: The cleaned dataframe. If an error occurs during the cleaning process, returns None.

    Raises:
    Exception: If an error occurs during the cleaning process.
    """
    try:
        messages_df = process_messages(messages_df)
        logging.info('Processing message data...')
        categories_df = process_categories(categories_df)
        logging.info('Processing category data..')
        categorized_messages_df = join_messages_and_categories(messages_df, categories_df)
        logging.info('Joining categories and messages...')
        categorized_messages_df = drop_ambigious_messages(categorized_messages_df)
        cleaned_df = drop_duplicates(categorized_messages_df)
    except Exception as e:
        logging.error("Error cleaning data: %s", e)
        raise

    return cleaned_df

def save_data(df, database_filename):
    """
    Save the DataFrame to a SQLite database.

    This function saves the DataFrame to a SQLite database at the specified path. The table name is the same as the database filename without the extension.

    Parameters:
    df (pandas.DataFrame): The DataFrame to save.
    database_filename (str): The path to the SQLite database.

    Returns:
    None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    table_name = os.path.splitext(os.path.basename(database_filename))[0]
    df.to_sql(table_name, engine, index=False, if_exists='replace')

def main():
    """
    Load data from the specified input files, clean the data, and save it to a database.

    The file paths are provided as command-line arguments. The first argument is the path to the messages file, 
    the second argument is the path to the categories file, and the third argument is the path to the database 
    where the cleaned data will be saved.

    If the correct number of arguments are not provided, an error message is logged and the function returns 
    without doing anything.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        logging.info('Loading data...\n    MESSAGES: %s\n    CATEGORIES: %s',
              messages_filepath, categories_filepath)
        messages_df, categories_df = load_data(messages_filepath, categories_filepath)

        logging.info('Cleaning data...')
        df = clean_data(messages_df, categories_df)

        logging.info('Saving data...\n    DATABASE: %s', database_filepath)
        save_data(df, database_filepath)

        logging.info('Cleaned data saved to database!')

    else:
        logging.error('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
