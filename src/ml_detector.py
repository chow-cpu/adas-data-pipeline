import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_PATH = "output/ml_model.pkl"
SCALER_PATH = "output/ml_scaler.pkl"

FEATURES = ["speed_mps", "accel_x", "accel_y", 
            "accel_z", "steering_angle", "radar_distance_m"]

def train_model(df):
    """
    Train an Isolation Forest model on normal driving data.
    Vehicle A highway data is used as the 'normal' baseline.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    model = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_estimators=100
    )
    model.fit(X)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Model trained and saved!")
    return model, scaler

def load_model():
    """Load a previously trained model."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def predict_anomalies(df, model, scaler):
    """
    Use the trained model to detect anomalies.
    Returns a copy of df with a new column 'ml_anomaly'
    True = anomaly, False = normal
    """
    X = scaler.transform(df[FEATURES])
    predictions = model.predict(X)
    df = df.copy()
    # Isolation Forest returns -1 for anomalies, 1 for normal
    df["ml_anomaly"] = predictions == -1
    df["ml_score"] = model.score_samples(X)
    return df

def get_anomaly_rows(df):
    """Return only the rows flagged as anomalies by the ML model."""
    return df[df["ml_anomaly"] == True]