import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import sys
sys.path.insert(0, "src")
from detect import detect_outliers
from alert_logger import log_alerts, get_log, clear_log
from ml_detector import train_model, load_model, predict_anomalies, get_anomaly_rows

st.set_page_config(page_title="ADAS Multi-Vehicle Dashboard", layout="wide")
st.title("ADAS Multi-Vehicle Sensor Dashboard")
st.markdown("Comparing 3 simultaneous vehicle simulations — C++ simulator + Python pipeline + ML")

# Load all three vehicles
df_a = pd.read_csv("data/vehicle_a.csv")
df_b = pd.read_csv("data/vehicle_b.csv")
df_c = pd.read_csv("data/vehicle_c.csv")

vehicles = {
    "Vehicle A — Highway": df_a,
    "Vehicle B — City": df_b,
    "Vehicle C — Aggressive": df_c,
}

vehicle_ids = {
    "Vehicle A — Highway": "VH-001",
    "Vehicle B — City": "VH-002",
    "Vehicle C — Aggressive": "VH-003",
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

# Top metrics
st.markdown("### Fleet Overview")
col1, col2, col3 = st.columns(3)
for col, (name, df) in zip([col1, col2, col3], vehicles.items()):
    with col:
        st.markdown(f"**{name}**")
        st.metric("Avg Speed (m/s)", f"{df['speed_mps'].mean():.2f}")
        st.metric("Max Speed (m/s)", f"{df['speed_mps'].max():.2f}")
        st.metric("Avg Radar (m)", f"{df['radar_distance_m'].mean():.2f}")

st.markdown("---")

# Controls
channel = st.selectbox("Select sensor channel:", ["speed_mps", "accel_x", "accel_y", "accel_z", "steering_angle", "radar_distance_m"])
threshold = st.slider("Anomaly detection threshold", 1.0, 3.0, 2.0, 0.1)

# ML Section
st.markdown("---")
st.markdown("### Machine Learning Anomaly Detection")
st.markdown("Train on Vehicle A (normal highway data) → detect anomalies in all vehicles")

col1, col2 = st.columns(2)
with col1:
    if st.button("Train ML Model on Vehicle A"):
        model, scaler = train_model(df_a)
        st.success("Model trained on Vehicle A highway data!")

with col2:
    model, scaler = load_model()
    if model is not None:
        st.info("Model loaded and ready")
    else:
        st.warning("No model trained yet — click Train first")

if model is not None:
    st.markdown("#### ML Detection Results")
    col1, col2, col3 = st.columns(3)

    for col, (name, df) in zip([col1, col2, col3], vehicles.items()):
        df_pred = predict_anomalies(df, model, scaler)
        ml_outliers = get_anomaly_rows(df_pred)
        with col:
            st.markdown(f"**{name}**")
            if len(ml_outliers) > 0:
                st.error(f"ML flagged {len(ml_outliers)} anomalies")
            else:
                st.success("ML: No anomalies")

    # ML vs Z-score comparison
    st.markdown("#### ML vs Z-Score Comparison")
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    for i, (name, df) in enumerate(vehicles.items()):
        df_pred = predict_anomalies(df, model, scaler)
        ml_outliers = get_anomaly_rows(df_pred)
        zscore_outliers = detect_outliers(df, channel, threshold)

        # Z-score row
        axes[0][i].plot(df["timestamp"], df[channel], color=colors[name])
        if len(zscore_outliers) > 0:
            axes[0][i].scatter(zscore_outliers["timestamp"], zscore_outliers[channel],
                             color="red", zorder=5, label=f"Z-score ({len(zscore_outliers)})")
        axes[0][i].set_title(f"{name}\nZ-Score Detection")
        axes[0][i].legend()

        # ML row
        axes[1][i].plot(df["timestamp"], df[channel], color=colors[name])
        if len(ml_outliers) > 0:
            axes[1][i].scatter(ml_outliers["timestamp"], ml_outliers[channel],
                             color="purple", zorder=5, label=f"ML ({len(ml_outliers)})")
        axes[1][i].set_title(f"{name}\nML Detection")
        axes[1][i].legend()

    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# Original sensor comparison
st.markdown("### Sensor Comparison (Z-Score)")
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 4))
for ax, (name, df) in zip(axes2, vehicles.items()):
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
st.pyplot(fig2)

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

# GPS map
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
            radius=3, color=color, fill=True,
            fill_opacity=0.6, tooltip=name
        ).add_to(m)
    for _, row in outliers.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8, color="red", fill=True,
            fill_opacity=0.9,
            popup=f"{name} Anomaly! {channel}: {row[channel]:.2f}"
        ).add_to(m)

st_folium(m, width=1400, height=500)

st.markdown("---")

# Alert log
st.markdown("### Alert History Log")
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("Clear Log"):
        clear_log()
        st.success("Log cleared!")

if st.button("Log Current Anomalies to Alert History"):
    for name, df in vehicles.items():
        outliers = detect_outliers(df, channel, threshold)
        log_alerts(outliers, vehicle_ids[name], channel)
    st.success("Alerts logged successfully!")

log = get_log()
if len(log) == 0:
    st.info("No alerts logged yet.")
else:
    severity_colors = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}
    log["severity"] = log["severity"].apply(lambda x: f"{severity_colors.get(x, '')} {x}")
    st.dataframe(log, use_container_width=True)
    st.caption(f"Total alerts logged: {len(log)}")

st.markdown("---")
st.markdown("### Raw Data")
tab1, tab2, tab3 = st.tabs(["Vehicle A", "Vehicle B", "Vehicle C"])
with tab1:
    st.dataframe(df_a)
with tab2:
    st.dataframe(df_b)
with tab3:
    st.dataframe(df_c)