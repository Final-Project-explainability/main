import os

import pandas as pd


def load_data():
    """
    Load dataset from the provided CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    print('Loading dataset...')
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_v2.csv')
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
