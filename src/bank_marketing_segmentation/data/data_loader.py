import pandas as pd

def load_data(file_path):
    """
    Load data from a file into a pandas DataFrame.

    Parameters:
    file_path (str): Path to the data file.

    Returns:
    pd.DataFrame: Loaded data.
    """
    data = pd.read_excel(file_path)
    return data
