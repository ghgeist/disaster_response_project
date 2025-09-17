"""
ETL pipeline for disaster response classification data.

This module handles the extraction, transformation, and loading of disaster
response data from raw CSV files into a cleaned, structured format suitable
for machine learning.
"""

import logging
import os
import pandas as pd
from sqlalchemy import create_engine


def load_raw_data(messages_filepath, categories_filepath):
    """
    Load raw disaster messages and categories data.
    
    Args:
        messages_filepath (str): Path to disaster_messages.csv
        categories_filepath (str): Path to disaster_categories.csv
        
    Returns:
        tuple: (messages_df, categories_df)
    """
    try:
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
        
        logging.info("Loaded %s messages and %s categories", len(messages), len(categories))
        return messages, categories
        
    except FileNotFoundError as e:
        logging.error("File not found: %s", e)
        raise
    except Exception as e:
        logging.error("Error loading raw data: %s", e)
        raise


def merge_messages_and_categories(messages_df, categories_df):
    """
    Merge messages and categories dataframes on id.
    
    Args:
        messages_df (pd.DataFrame): Messages dataframe
        categories_df (pd.DataFrame): Categories dataframe
        
    Returns:
        pd.DataFrame: Merged dataframe
    """
    try:
        df = messages_df.merge(categories_df, on='id')
        logging.info("Merged dataframes: %s", df.shape)
        return df
        
    except Exception as e:
        logging.error("Error merging dataframes: %s", e)
        raise


def split_categories_column(df):
    """
    Split the categories column into separate binary columns.
    
    Args:
        df (pd.DataFrame): Dataframe with categories column
        
    Returns:
        pd.DataFrame: Dataframe with split category columns
    """
    try:
        # Split the categories column into separate columns
        split_categories = df['categories'].str.split(';', expand=True)
        
        # Create a list of column names for categories
        category_colnames = [x.split('-')[0] for x in split_categories.iloc[0, :]]
        
        # Rename the columns of categories
        split_categories.columns = category_colnames
        
        # Concatenate the 'id' column with the split categories
        result = pd.concat([df['id'], split_categories], axis=1)
        
        logging.info("Split categories into %s columns", len(category_colnames))
        return result
        
    except Exception as e:
        logging.error("Error splitting categories: %s", e)
        raise


def convert_categories_to_numeric(df):
    """
    Convert category columns from string format to numeric.
    
    Args:
        df (pd.DataFrame): Dataframe with string category columns
        
    Returns:
        pd.DataFrame: Dataframe with numeric category columns
    """
    try:
        for column in df:
            if column == 'id':
                continue
            # Set each value to be the last character of the string
            df.loc[:, column] = df[column].str[-1]
            # Convert column from string to numeric
            df.loc[:, column] = pd.to_numeric(df[column])
            
        logging.info("Converted category columns to numeric")
        return df
        
    except Exception as e:
        logging.error("Error converting categories to numeric: %s", e)
        raise


def clean_related_column(df):
    """
    Clean the 'related' column by removing ambiguous values (2).
    
    Args:
        df (pd.DataFrame): Dataframe with 'related' column
        
    Returns:
        pd.DataFrame: Dataframe with cleaned 'related' column
    """
    try:
        # Check distribution before cleaning
        related_counts = df['related'].value_counts()
        logging.info("Related column distribution before cleaning: %s", related_counts.to_dict())
        
        # Drop rows with value 2 (ambiguous)
        original_count = len(df)
        df = df[df['related'] != 2]
        removed_count = original_count - len(df)
        
        if removed_count > 0:
            logging.warning("Removed %s rows with ambiguous 'related' values", removed_count)
            
        logging.info("Cleaned 'related' column: %s rows remaining", len(df))
        return df
        
    except Exception as e:
        logging.error("Error cleaning 'related' column: %s", e)
        raise


def remove_duplicates(df):
    """
    Remove duplicate rows from the dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe to deduplicate
        
    Returns:
        pd.DataFrame: Deduplicated dataframe
    """
    try:
        original_count = len(df)
        duplicates_count = df.duplicated().sum()
        
        if duplicates_count > 0:
            df = df.drop_duplicates()
            removed_count = original_count - len(df)
            logging.warning("Removed %s duplicate rows", removed_count)
        else:
            logging.info("No duplicates found")
            
        logging.info("Final dataframe shape: %s", df.shape)
        return df
        
    except Exception as e:
        logging.error("Error removing duplicates: %s", e)
        raise


def save_processed_data(df, csv_filepath, db_filepath, table_name='stg_disaster_response'):
    """
    Save processed data to CSV and SQLite database.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        csv_filepath (str): Path to save CSV file
        db_filepath (str): Path to SQLite database
        table_name (str): Name of the table in the database
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(db_filepath), exist_ok=True)
        
        # Save to CSV
        df.to_csv(csv_filepath, index=False)
        logging.info("Saved data to CSV: %s", csv_filepath)
        
        # Save to SQLite database
        engine = create_engine(f'sqlite:///{db_filepath}')
        df.to_sql(table_name, engine, index=False, if_exists='replace')
        logging.info("Saved data to database: %s", db_filepath)
        
    except Exception as e:
        logging.error("Error saving processed data: %s", e)
        raise


def run_etl_pipeline(messages_filepath, categories_filepath, output_csv_path, output_db_path):
    """
    Run the complete ETL pipeline.
    
    Args:
        messages_filepath (str): Path to disaster_messages.csv
        categories_filepath (str): Path to disaster_categories.csv
        output_csv_path (str): Path to save processed CSV
        output_db_path (str): Path to save SQLite database
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    try:
        logging.info("Starting ETL pipeline")
        
        # Extract
        messages_df, categories_df = load_raw_data(messages_filepath, categories_filepath)
        
        # Transform
        df = merge_messages_and_categories(messages_df, categories_df)
        
        # Split categories
        categories_result = split_categories_column(df)
        categories_result = convert_categories_to_numeric(categories_result)
        
        # Clean data
        categories_result = clean_related_column(categories_result)
        
        # Merge back with original data
        df = df.drop('categories', axis=1)
        df = df.merge(categories_result, on='id')
        
        # Remove duplicates
        df = remove_duplicates(df)
        
        # Load
        save_processed_data(df, output_csv_path, output_db_path)
        
        logging.info("ETL pipeline completed successfully")
        return df
        
    except Exception as e:
        logging.error("ETL pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    messages_file = "data/01_raw/disaster_messages.csv"
    categories_file = "data/01_raw/disaster_categories.csv"
    output_csv = "data/02_stg/stg_disaster_messages.csv"
    output_db = "data/02_stg/stg_disaster_response.db"
    
    df = run_etl_pipeline(messages_file, categories_file, output_csv, output_db)
    print(f"ETL pipeline completed. Processed {len(df)} records.")
