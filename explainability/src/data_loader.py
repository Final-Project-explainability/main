import pandas as pd


def load_data(file_path):
    """
    Load dataset from the provided CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    print('Loading dataset...')
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
