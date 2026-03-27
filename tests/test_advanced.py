import sys
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
import pytest
from detect import detect_outliers
from preprocess import clean, normalize_column
from ingest import load_csv

# ================================================
# OPTION 1 — Stress test with large data
# ================================================

def make_large_dataset(n=10000):
    """Generate a large dataset with one anomaly."""
    np.random.seed(42)
    data = {
        "timestamp": [i * 0.1 for i in range(n)],
        "speed_mps": np.random.normal(10.0, 0.3, n).tolist(),
        "accel_x": np.random.normal(0.2, 0.1, n).tolist(),
        "accel_y": np.random.normal(-0.1, 0.05, n).tolist(),
        "accel_z": np.random.normal(9.8, 0.1, n).tolist(),
        "steering_angle": np.random.normal(2.0, 0.5, n).tolist(),
        "radar_distance_m": np.random.normal(45.0, 1.0, n).tolist(),
    }
    # Plant one obvious anomaly
    data["speed_mps"][5000] = 99.0
    return pd.DataFrame(data)

def test_stress_large_dataset_loads():
    """Pipeline should handle 10,000 rows without crashing."""
    df = make_large_dataset(10000)
    df = clean(df)
    assert len(df) == 10000

def test_stress_large_dataset_detects_anomaly():
    """Anomaly detector should find the planted spike in 10,000 rows."""
    df = make_large_dataset(10000)
    outliers = detect_outliers(df, "speed_mps", threshold=2.0)
    assert len(outliers) >= 1

def test_stress_normalization_large():
    """Normalization should work correctly on large datasets."""
    df = make_large_dataset(10000)
    df = normalize_column(df, "speed_mps")
    assert df["speed_mps_normalized"].min() >= 0.0
    assert df["speed_mps_normalized"].max() <= 1.0

# ================================================
# OPTION 2 — Edge case testing
# ================================================

def test_edge_missing_values_cleaned():
    """Rows with missing values should be removed by clean()."""
    df = pd.DataFrame({
        "timestamp": [0, 1, 2, 3],
        "speed_mps": [10.0, None, 10.2, 10.1],
        "accel_x": [0.1, 0.2, None, 0.1],
        "accel_y": [0.0, 0.0, 0.0, 0.0],
        "accel_z": [9.8, 9.8, 9.8, 9.8],
        "steering_angle": [2.0, 2.0, 2.0, 2.0],
        "radar_distance_m": [45.0, 45.0, 45.0, 45.0],
    })
    cleaned = clean(df)
    assert len(cleaned) == 2

def test_edge_negative_speed():
    """Detector should flag negative speed as anomaly."""
    df = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "speed_mps": [10.0, 10.1, 10.2, 10.1, -50.0]
    })
    outliers = detect_outliers(df, "speed_mps", threshold=1.0)
    assert len(outliers) == 1

def test_edge_all_identical_values():
    """Detector should handle all identical values without crashing."""
    df = pd.DataFrame({
        "timestamp": [0, 1, 2, 3, 4],
        "speed_mps": [10.0, 10.0, 10.0, 10.0, 10.0]
    })
    # Should not crash even with zero std deviation
    try:
        outliers = detect_outliers(df, "speed_mps")
        assert True
    except Exception as e:
        pytest.fail(f"Detector crashed on identical values: {e}")

def test_edge_single_row():
    """Pipeline should handle a single row dataset."""
    df = pd.DataFrame({
        "timestamp": [0],
        "speed_mps": [10.0],
        "accel_x": [0.1],
        "accel_y": [0.0],
        "accel_z": [9.8],
        "steering_angle": [2.0],
        "radar_distance_m": [45.0],
    })
    cleaned = clean(df)
    assert len(cleaned) == 1

# ================================================
# OPTION 3 — Threshold sensitivity testing
# ================================================

def make_sensitivity_data():
    """Dataset with one extreme outlier."""
    np.random.seed(0)
    speeds = np.random.normal(10.0, 0.5, 50).tolist()
    speeds[25] = 99.0
    return pd.DataFrame({
        "timestamp": [i * 0.1 for i in range(50)],
        "speed_mps": speeds
    })

def test_threshold_low_catches_more():
    """Lower threshold should catch more anomalies."""
    df = make_sensitivity_data()
    outliers_strict = detect_outliers(df, "speed_mps", threshold=3.0)
    outliers_loose = detect_outliers(df, "speed_mps", threshold=1.0)
    assert len(outliers_loose) >= len(outliers_strict)

def test_threshold_high_catches_fewer():
    """Very high threshold should catch fewer or no anomalies."""
    df = make_sensitivity_data()
    outliers = detect_outliers(df, "speed_mps", threshold=10.0)
    assert len(outliers) == 0

def test_threshold_extreme_outlier_always_caught():
    """An extreme outlier should be caught at any reasonable threshold."""
    df = make_sensitivity_data()
    for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
        outliers = detect_outliers(df, "speed_mps", threshold=threshold)
        assert len(outliers) >= 1, f"Missed outlier at threshold {threshold}"

# ================================================
# OPTION 4 — Multiple driving scenarios
# ================================================

def make_scenario(scenario="highway"):
    """Generate sensor data for different driving scenarios."""
    np.random.seed(1)
    n = 100

    if scenario == "highway":
        speeds = np.random.normal(30.0, 0.5, n)
        radar = np.random.normal(80.0, 2.0, n)
    elif scenario == "city":
        speeds = np.random.normal(8.0, 1.0, n)
        radar = np.random.normal(15.0, 3.0, n)
    elif scenario == "emergency_brake":
        speeds = np.random.normal(25.0, 0.5, n)
        speeds[50] = 0.5
        radar = np.random.normal(5.0, 1.0, n)

    return pd.DataFrame({
        "timestamp": [i * 0.1 for i in range(n)],
        "speed_mps": speeds,
        "radar_distance_m": radar
    })

def test_scenario_highway_high_speed():
    """Highway scenario should have average speed above 25 m/s."""
    df = make_scenario("highway")
    assert df["speed_mps"].mean() > 25.0

def test_scenario_city_low_speed():
    """City scenario should have average speed below 15 m/s."""
    df = make_scenario("city")
    assert df["speed_mps"].mean() < 15.0

def test_scenario_emergency_brake_detected():
    """Emergency brake scenario should trigger anomaly detection."""
    df = make_scenario("emergency_brake")
    outliers = detect_outliers(df, "speed_mps", threshold=2.0)
    assert len(outliers) >= 1

def test_scenario_city_close_radar():
    """City scenario should have closer radar readings than highway."""
    city = make_scenario("city")
    highway = make_scenario("highway")
    assert city["radar_distance_m"].mean() < highway["radar_distance_m"].mean()