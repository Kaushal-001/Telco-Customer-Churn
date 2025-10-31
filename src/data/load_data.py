import pandas as pd
import os

def load_data(file_path : str) -> pd.DataFrame:
    """
    Load data from a csv file into a pandas dataframe.

    Args:
        file_path (str): The path to the csv file.
    
    Returns:
        pd.DataFrame: The loaded data as a pandas dataframe.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path)

