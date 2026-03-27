def detect_outliers(df, column, threshold=2.0):
    """
    Flag rows where the value is more than `threshold`
    standard deviations from the mean.
    Returns a new DataFrame with only the flagged rows.
    """
    mean = df[column].mean()
    std = df[column].std()
    outliers = df[abs(df[column] - mean) > threshold * std]
    print(f"Found {len(outliers)} outlier(s) in column '{column}'")
    return outliers 
