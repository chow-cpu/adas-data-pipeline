import pandas as pd

def load_csv(filepath):
    """Load a sensor log CSV file and return a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    return df