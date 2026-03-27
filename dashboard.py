import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import sys
sys.path.insert(0, "src")
from detect import detect_outliers

st.set_page_config(page_title="ADAS Multi-Vehicle Dashboard", layout="wide")
st.title("ADAS Multi-Vehicle Sensor Dashboard")
st.markdown("Comparing 3 simultaneous vehicle simulations — C++ simulator + Python pipeline")

# Load all three vehicles
df_a = pd.read_csv("data/vehicle_a.csv")
df_b = pd.read_csv("data/vehicle_b.csv")
df_c = pd.read_csv("data/vehicle_c.csv")

vehicles = {
    "Vehicle A — Highway": df_a,
    "Vehicle B — City": df_b,
    "Vehicle C — Aggressive": df_c,
}

colors = {
    "Vehicle A — Highway": "steelblue",
    "Vehicle B — City": "green",
    "Vehicle C — Aggressive": "orange",
}

map_colors = {
    "Vehicle A — Highway": "blue",
    "Vehicle B — City": "green",
    "Vehicle C — Aggressive": "orange",
}

# Top metrics for each vehicle
st.markdown("### Fleet Overview")
col1, col2, col3 = st.columns(3)

for col, (name, df) in zip([col1, col2, col3], vehicles.items()):
    with col:
        st.markdown(f"**{name}**")
        st.metric("Avg Speed (m/s)", f"{df['speed_mps'].mean():.2f}")
        st.metric("Max Speed (m/s)", f"{df['speed_mps'].max():.2f}")
        st.metric("Avg Radar (m)", f"{df['radar_distance_m'].mean():.2f}")

st.markdown("---")

# Channel selector and threshold
channel = st.selectbox("Select sensor channel:", ["speed_mps", "accel_x", "accel_y", "accel_z", "steering_angle", "radar_distance_m"])
threshold = st.slider("Anomaly detection threshold", 1.0, 3.0, 2.0, 0.1)

# Side by side charts
st.markdown("### Sensor Comparison")
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

for ax, (name, df) in zip(axes, vehicles.items()):
    outliers = detect_outliers(df, channel, threshold)
    ax.plot(df["timestamp"], df[channel], color=colors[name], label=channel)
    if len(outliers) > 0:
        ax.scatter(outliers["timestamp"], outliers[channel],
                   color="red", zorder=5, label=f"Anomalies ({len(outliers)})")
    ax.set_title(name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(channel)
    ax.legend()

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# Anomaly summary
st.markdown("### Anomaly Summary")
col1, col2, col3 = st.columns(3)

for col, (name, df) in zip([col1, col2, col3], vehicles.items()):
    outliers = detect_outliers(df, channel, threshold)
    with col:
        if len(outliers) > 0:
            st.error(f"{name}: {len(outliers)} anomaly found")
        else:
            st.success(f"{name}: No anomalies")

st.markdown("---")

# GPS map with all three vehicles
st.markdown("### GPS Route Map — All Vehicles")
st.markdown("🔵 Vehicle A (Highway) | 🟢 Vehicle B (City) | 🟠 Vehicle C (Aggressive) | 🔴 Anomalies")

center_lat = df_a["latitude"].mean()
center_lon = df_a["longitude"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

for name, df in vehicles.items():
    outliers = detect_outliers(df, channel, threshold)
    color = map_colors[name]

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.6,
            tooltip=name
        ).add_to(m)

    for _, row in outliers.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color="red",
            fill=True,
            fill_opacity=0.9,
            popup=f"{name} Anomaly! {channel}: {row[channel]:.2f}"
        ).add_to(m)

st_folium(m, width=1400, height=500)

st.markdown("---")
st.markdown("### Raw Data")
tab1, tab2, tab3 = st.tabs(["Vehicle A", "Vehicle B", "Vehicle C"])
with tab1:
    st.dataframe(df_a)
with tab2:
    st.dataframe(df_b)
with tab3:
    st.dataframe(df_c)