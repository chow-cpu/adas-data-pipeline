import sys
sys.path.insert(0, "src")

import pandas as pd
from ingest import load_csv

def test_load_csv_returns_dataframe():
    df = load_csv("data/sample_run.csv")
    assert isinstance(df, pd.DataFrame)

def test_load_csv_has_correct_columns():
    df = load_csv("data/sample_run.csv")
    assert "speed_mps" in df.columns
    assert "timestamp" in df.columns

def test_load_csv_not_empty():
    df = load_csv("data/sample_run.csv")
    assert len(df) > 0 
