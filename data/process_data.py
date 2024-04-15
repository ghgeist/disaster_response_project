#Standard Imports
import logging
import os
import sys

#Third-party Imports
import pandas as pd
from sqlalchemy import create_engine

# Set up logging
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler('app.log')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

rootLogger.setLevel(logging.INFO)

def load_data(messages_filepath, categories_filepath):
    """
    Load data from CSV files.

    This function reads the messages and categories data from CSV files, merges them into a single DataFrame,
    and returns the merged DataFrame.

    Parameters:
    messages_filepath (str): The path to the messages CSV file.
    categories_filepath (str): The path to the categories CSV file.

    Returns:
    df (pandas.DataFrame): The merged DataFrame.

    If an error occurs while loading the data, None is returned.
    """
    try:
        # Load messages dataset
        messages = pd.read_csv(messages_filepath)
        # Load categories dataset
        categories = pd.read_csv(categories_filepath)
        # Merge datasets
        df = pd.merge(messages, categories, on='id')
    except Exception as e:
        logging.error("Error loading data: %s", e)
        return None

    return df


def clean_data(df):
    """
    Clean the data in the DataFrame.

    This function splits the 'categories' column into separate columns, converts the category values to just numbers 0 or 1,
    drops the original 'categories' column, and removes duplicates.

    Parameters:
    df (pandas.DataFrame): The DataFrame to clean.

    Returns:
    df (pandas.DataFrame): The cleaned DataFrame.

    If an error occurs while cleaning the data, None is returned and the error is logged.
    """
    try:
        # Split the categories column into separate columns
        split_categories = df['categories'].str.split(';', expand=True)

        # Create a list of column names for categories
        category_colnames = [x.split('-')[0] for x in split_categories.iloc[0, :]]

        # Rename the columns of `categories`
        split_categories.columns = category_colnames

        # Convert category values to just numbers 0 or 1
        for column in split_categories:
            # set each value to be the last character of the string
            split_categories[column] = split_categories[column].str[-1]
            # convert column from string to numeric
            split_categories[column] = pd.to_numeric(split_categories[column])

        # Drop the original categories column from `df`
        df = df.drop('categories', axis=1)

        # Join the split categories with the original dataframe
        df = pd.concat([df, split_categories], axis=1)

        # Drop duplicates
        df = df.drop_duplicates()

    except Exception as e:
        logging.error("Error cleaning data: %s", e)
        return None

    return df

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
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        logging.info('Loading data...\n    MESSAGES: %s\n    CATEGORIES: %s',
              messages_filepath, categories_filepath)
        df = load_data(messages_filepath, categories_filepath)

        logging.info('Cleaning data...')
        df = clean_data(df)

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
    logging.basicConfig(level=logging.INFO)
    main()