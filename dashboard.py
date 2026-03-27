import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "src")
from detect import detect_outliers

st.set_page_config(page_title="ADAS Sensor Dashboard", layout="wide")
st.title("ADAS Sensor Data Dashboard")
st.markdown("Real-time vehicle sensor analysis — C++ simulator + Python pipeline")

df = pd.read_csv("data/simulated_sensor_log.csv")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Readings", len(df))
col2.metric("Avg Speed (m/s)", f"{df['speed_mps'].mean():.2f}")
col3.metric("Max Speed (m/s)", f"{df['speed_mps'].max():.2f}")
col4.metric("Avg Radar (m)", f"{df['radar_distance_m'].mean():.2f}")

st.markdown("---")

channel = st.selectbox("Select sensor channel:", ["speed_mps", "accel_x", "accel_y", "accel_z", "steering_angle", "radar_distance_m"])
threshold = st.slider("Anomaly detection threshold", 1.0, 3.0, 2.0, 0.1)

outliers = detect_outliers(df, channel, threshold)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["timestamp"], df[channel], color="steelblue", label=channel)
if len(outliers) > 0:
    ax.scatter(outliers["timestamp"], outliers[channel], color="red", zorder=5, label=f"Anomalies ({len(outliers)})")
ax.set_xlabel("Time (s)")
ax.set_ylabel(channel)
ax.legend()
st.pyplot(fig)

st.markdown("---")
if len(outliers) > 0:
    st.error(f"Found {len(outliers)} anomaly in '{channel}'")
    st.dataframe(outliers)
else:
    st.success(f"No anomalies detected in '{channel}'")

st.markdown("---")
st.subheader("Raw Sensor Data")
st.dataframe(df)