import pandas as pd

def check_nulls_in_columns(data, column_type):
    """
    Check and print if there are nulls or 'unknown' values in specified types of columns.

    Parameters:
    data (pd.DataFrame): The DataFrame to check.
    column_type (str or list): Type of columns to check (e.g., 'object', ['int64', 'float64']).
    """
    columns = data.select_dtypes(include=column_type).columns
    for col in columns:
        if data[col].isnull().any() or (data[col] == 'unknown').any():
            print(f"There are nulls or 'unknown' in {col}")
        else:
            print(f"No Nulls in {col}")

def print_unique_values(data, column_type):
    """
    Print unique values in specified types of columns.

    Parameters:
    data (pd.DataFrame): The DataFrame to process.
    column_type (str): Type of columns to process (e.g., 'object').
    """
    columns = data.select_dtypes(include=column_type).columns
    for col in columns:
        print(f"{col}: {data[col].unique()}")

def count_and_print_unknowns(data, columns):
    """
    Count and print the number of 'unknown' values in specified columns.

    Parameters:
    data (pd.DataFrame): The DataFrame to process.
    columns (list): List of column names to process.
    """
    for col in columns:
        unknown_count = (data[col] == 'unknown').sum()
        print(f"{col}: {data[col].unique()}, Number of 'unknown': {unknown_count}")

def drop_columns(data, columns_to_drop):
    """
    Drop specified columns from the DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame to process.
    columns_to_drop (list): List of column names to drop.

    Returns:
    pd.DataFrame: The DataFrame with specified columns dropped.
    """
    return data.drop(columns=columns_to_drop)

def filter_rows(data, filter_conditions):
    """
    Filter out rows based on specified conditions.

    Parameters:
    data (pd.DataFrame): The DataFrame to process.
    filter_conditions (dict): Conditions to apply for filtering. 
                              Format: {'column_name': ['values to filter out']}

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    for col, values in filter_conditions.items():
        data = data[~data[col].isin(values)]
    return data

def preprocess_data(data):
    """
    Main function to preprocess data.

    Parameters:
    data (pd.DataFrame): The DataFrame to process.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    # Checking nulls
    check_nulls_in_columns(data, 'object')
    check_nulls_in_columns(data, ['int64', 'float64'])

    # Printing unique values in categorical columns
    print_unique_values(data, 'object')

    # Counting and printing unknowns in specified columns
    columns_with_unknowns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'pdays']
    count_and_print_unknowns(data, columns_with_unknowns)

    # Dropping specified columns
    columns_to_drop = ['duration', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    data = drop_columns(data, columns_to_drop)

    # Filtering rows based on conditions
    filter_conditions = {'default': ['yes'], 'job': ['unknown'], 'marital': ['unknown']}
    data = filter_rows(data, filter_conditions)

    return data

# Example usage
# bank_data = pd.read_excel('path/to/bank_data.xlsx')
# preprocessed_data = preprocess_data(bank_data)
