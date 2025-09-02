"""
Column definitions for disaster response classification data.

This module contains the definitions and metadata for all columns
in the disaster response dataset.
"""

import pandas as pd
import logging


def get_column_definitions():
    """
    Get the column definitions dictionary.
    
    Returns:
        dict: Dictionary mapping column names to their definitions
    """
    return {
        'id': 'Unique identifier for each message',
        'message': 'Content of the message',
        'original': 'Original message text (if available). Given that the messages are in Haitian Creole, this column is useful for translation purposes',
        'genre': 'Genre of the message. Values should be news, direct and social',
        'related': 'Whether the message is related to the disaster or not. 0 means not related, 1 means related, 2 means unclassifiable',
        'request': 'Whether the message is a request for help (binary)',
        'offer': 'Whether the message is an offer of help (binary)',
        'aid_related': 'Whether the message is related to aid (binary)',
        'medical_help': 'Whether the message requests medical help (binary)',
        'medical_products': 'Whether the message requests medical products (binary)',
        'search_and_rescue': 'Whether the message requests search and rescue help (binary)',
        'security': 'Whether the message requests security assistance (binary)',
        'military': 'Whether the message requests military assistance (binary)',
        'child_alone': 'Whether the message requests assistance for children alone (binary)',
        'water': 'Whether the message requests water (binary)',
        'food': 'Whether the message requests food (binary)',
        'shelter': 'Whether the message requests shelter (binary)',
        'clothing': 'Whether the message requests clothing (binary)',
        'money': 'Whether the message requests money (binary)',
        'missing_people': 'Whether the message reports missing people (binary)',
        'refugees': 'Whether the message reports refugees (binary)',
        'death': 'Whether the message reports deaths (binary)',
        'other_aid': 'Whether the message requests other aid (binary)',
        'infrastructure_related': 'Whether the message relates to infrastructure issues (binary)',
        'transport': 'Whether the message requests transportation assistance (binary)',
        'buildings': 'Whether the message reports damage to buildings (binary)',
        'electricity': 'Whether the message requests electricity (binary)',
        'tools': 'Whether the message requests tools (binary)',
        'hospitals': 'Whether the message requests hospitals (binary)',
        'shops': 'Whether the message requests shops (binary)',
        'aid_centers': 'Whether the message requests aid centers (binary)',
        'other_infrastructure': 'Whether the message requests other infrastructure assistance (binary)',
        'weather_related': 'Whether the message is related to weather events (binary)',
        'floods': 'Whether the message is related to floods (binary)',
        'storm': 'Whether the message is related to storms (binary)',
        'fire': 'Whether the message is related to fires (binary)',
        'earthquake': 'Whether the message is related to earthquakes (binary)',
        'cold': 'Whether the message is related to cold weather (binary)',
        'other_weather': 'Whether the message is related to other weather events (binary)',
        'direct_report': 'Whether the message is a direct report (binary)'
    }


def get_column_definitions_dataframe():
    """
    Get column definitions as a pandas DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with columns 'columns' and 'definition'
    """
    definitions = get_column_definitions()
    return pd.DataFrame(definitions.items(), columns=['columns', 'definition'])


def get_target_columns():
    """
    Get the list of target columns for classification (excluding metadata columns).
    
    Returns:
        list: List of target column names
    """
    # Exclude metadata columns and the 'related' column (which is used for filtering)
    metadata_columns = {'id', 'message', 'original', 'genre', 'related'}
    all_columns = set(get_column_definitions().keys())
    return sorted(list(all_columns - metadata_columns))


def get_boolean_columns():
    """
    Get the list of boolean/categorical columns.
    
    Returns:
        list: List of boolean column names
    """
    return get_target_columns()


def get_string_columns():
    """
    Get the list of string/text columns.
    
    Returns:
        list: List of string column names
    """
    return ['message', 'original', 'genre']


def get_integer_columns():
    """
    Get the list of integer columns.
    
    Returns:
        list: List of integer column names
    """
    return ['id', 'related']


def get_data_types():
    """
    Get a dictionary mapping column names to their data types.
    
    Returns:
        dict: Dictionary mapping column names to pandas data types
    """
    dtype_dict = {}
    
    # Boolean columns
    for col in get_boolean_columns():
        dtype_dict[col] = bool
    
    # String columns
    for col in get_string_columns():
        dtype_dict[col] = str
    
    # Integer columns
    for col in get_integer_columns():
        dtype_dict[col] = int
    
    return dtype_dict


def save_column_definitions(output_filepath):
    """
    Save column definitions to a CSV file.
    
    Args:
        output_filepath (str): Path to save the definitions CSV file
    """
    try:
        import os
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        definitions_df = get_column_definitions_dataframe()
        definitions_df.to_csv(output_filepath, index=False)
        
        logging.info(f"Column definitions saved to: {output_filepath}")
        
    except Exception as e:
        logging.error(f"Error saving column definitions: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Print column definitions
    definitions = get_column_definitions()
    print("Column Definitions:")
    for col, definition in definitions.items():
        print(f"  {col}: {definition}")
    
    # Print target columns
    target_cols = get_target_columns()
    print(f"\nTarget Columns ({len(target_cols)}): {target_cols}")
    
    # Save to file
    save_column_definitions("data/02_stg/stg_disaster_messages_definitions.csv")
