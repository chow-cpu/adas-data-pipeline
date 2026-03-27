def clean(df):
    """Remove missing values and reset the index."""
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(f"Data cleaned. {len(df)} rows remaining.")
    return df

def normalize_column(df, column):
    """Scale a column to a 0-1 range."""
    min_val = df[column].min()
    max_val = df[column].max()
    df[column + "_normalized"] = (df[column] - min_val) / (max_val - min_val)
    return df 
