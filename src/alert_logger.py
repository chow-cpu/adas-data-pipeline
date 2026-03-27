import pandas as pd
import os
from datetime import datetime

LOG_PATH = "output/alert_log.csv"

def init_log():
    """Create the log file with headers if it doesn't exist."""
    if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        df = pd.DataFrame(columns=[
            "logged_at", "vehicle_id", "channel",
            "value", "severity", "latitude", "longitude"
        ])
        df.to_csv(LOG_PATH, index=False)

def log_alerts(outliers, vehicle_id, channel):
    """Save detected anomalies to the alert log."""
    init_log()
    if len(outliers) == 0:
        return

    existing = pd.read_csv(LOG_PATH)
    mean = outliers[channel].mean()
    new_rows = []

    for _, row in outliers.iterrows():
        deviation = abs(row[channel] - mean)
        if deviation < 2:
            severity = "LOW"
        elif deviation < 5:
            severity = "MEDIUM"
        else:
            severity = "HIGH"

        new_rows.append({
            "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "vehicle_id": vehicle_id,
            "channel": channel,
            "value": round(row[channel], 4),
            "severity": severity,
            "latitude": round(row["latitude"], 6),
            "longitude": round(row["longitude"], 6),
        })

    updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    updated.to_csv(LOG_PATH, index=False)
    print(f"Logged {len(new_rows)} alert(s) for {vehicle_id}")

def get_log():
    """Read and return the full alert log."""
    init_log()
    return pd.read_csv(LOG_PATH)

def clear_log():
    """Clear the alert log."""
    init_log()
    df = pd.DataFrame(columns=[
        "logged_at", "vehicle_id", "channel",
        "value", "severity", "latitude", "longitude"
    ])
    df.to_csv(LOG_PATH, index=False)
