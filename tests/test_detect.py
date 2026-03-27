import sys
sys.path.insert(0, "src")

import pandas as pd
from detect import detect_outliers

def make_test_data():
    return pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "speed_mps": [10.0, 10.2, 10.1, 10.3, 99.0]
    })

def test_outlier_is_detected():
    df = make_test_data()
    outliers = detect_outliers(df, "speed_mps", threshold=1.0)
    assert len(outliers) == 1

def test_normal_data_has_no_outliers():
    df = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "speed_mps": [10.0, 10.1, 10.2, 10.1, 10.0]
    })
    outliers = detect_outliers(df, "speed_mps")
    assert len(outliers) == 0