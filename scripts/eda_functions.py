import pandas as pd

def summarize_data(df):
    # Check if df is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input should be a pandas DataFrame")

    # Check if df is empty
    if df.empty:
        raise ValueError("Input DataFrame should not be empty")

    # Calculate unique_count and missing_count in one pass
    summary_df = df.agg(['nunique', 'count', lambda x: x.isnull().sum()]).transpose()

    # Rename the columns
    summary_df.columns = ['unique_values', 'total_values', 'missing_values']
    
    # Calculate the percentage of missing values and round to two decimal places
    summary_df['missing_percentage'] = round((summary_df['missing_values'] / len(df)) * 100, 2)
    
    #Calculate the percentage of unique values and round to two decimal places
    summary_df['unique_percentage'] = round((summary_df['unique_values'] / summary_df['total_values']) * 100, 2)
    
    #Re-order the columns. 
    #The order should be column name, unique value count, uniuqe value percentage, missing_values, missing_percentage, and type
    summary_df = summary_df[['unique_values', 'unique_percentage', 'missing_values', 'missing_percentage']]

    # Add column types
    summary_df['type'] = df.dtypes

    # Reset the index
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={'index': 'columns'}, inplace=True)

    # Return the DataFrame
    return summary_df