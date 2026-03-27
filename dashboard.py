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
from health_score import calculate_health_score
from report_generator import generate_report
import time
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
model, scaler = load_model()

for col, (name, df) in zip([col1, col2, col3], vehicles.items()):
    with col:
        st.markdown(f"**{name}**")
        st.metric("Avg Speed (m/s)", f"{df['speed_mps'].mean():.2f}")
        st.metric("Max Speed (m/s)", f"{df['speed_mps'].max():.2f}")
        st.metric("Avg Radar (m)", f"{df['radar_distance_m'].mean():.2f}")

        if model is not None:
            df_pred = predict_anomalies(df, model, scaler)
            ml_outliers = get_anomaly_rows(df_pred)
            zscore_outliers = detect_outliers(df, "speed_mps", 2.0)
            score, status, color = calculate_health_score(df, ml_outliers, zscore_outliers)

            st.markdown(f"**Health Score**")
            if color == "green":
                st.success(f"✅ {score}/100 — {status}")
            elif color == "orange":
                st.warning(f"⚠️ {score}/100 — {status}")
            else:
                st.error(f"🔴 {score}/100 — {status}")
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

# PDF Report Generator
st.markdown("### Generate PDF Report")
if st.button("Generate Full PDF Report"):
    if model is not None:
        with st.spinner("Generating report..."):
            health_scores = {}
            zscore_outliers_dict = {}
            ml_outliers_dict = {}

            for name, df in vehicles.items():
                df_pred = predict_anomalies(df, model, scaler)
                ml_out = get_anomaly_rows(df_pred)
                z_out = detect_outliers(df, channel, threshold)
                score, status, color = calculate_health_score(df, ml_out, z_out)
                health_scores[name] = (score, status, color)
                zscore_outliers_dict[name] = z_out
                ml_outliers_dict[name] = ml_out

            path = generate_report(
                vehicles, vehicle_ids, health_scores,
                zscore_outliers_dict, ml_outliers_dict, channel
            )
            with open(path, "rb") as f:
                st.download_button(
                    label="Download PDF Report",
                    data=f,
                    file_name="adas_report.pdf",
                    mime="application/pdf"
                )
            st.success("Report generated successfully!")
    else:
        st.warning("Please train the ML model first!")

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
    st.markdown("---")

# Simulation Replay Mode
st.markdown("### Simulation Replay Mode")
st.markdown("Watch sensor data build frame by frame — like a live vehicle test bench")

replay_vehicle = st.selectbox(
    "Select vehicle to replay:",
    list(vehicles.keys()),
    key="replay_vehicle"
)

replay_channel = st.selectbox(
    "Select channel to replay:",
    ["speed_mps", "accel_x", "accel_y", "accel_z", "steering_angle", "radar_distance_m"],
    key="replay_channel"
)

replay_speed = st.select_slider(
    "Replay speed:",
    options=["0.25x", "0.5x", "1x", "2x", "4x"],
    value="1x"
)

speed_map = {"0.25x": 0.4, "0.5x": 0.2, "1x": 0.1, "2x": 0.05, "4x": 0.025}
delay = speed_map[replay_speed]

col1, col2 = st.columns(2)
play = col1.button("Play Replay")
stop = col2.button("Stop")

if play:
    df_replay = vehicles[replay_vehicle].copy()
    zscore_out = detect_outliers(df_replay, replay_channel, 2.0)
    anomaly_timestamps = set(zscore_out["timestamp"].values)

    chart_placeholder = st.empty()
    map_placeholder = st.empty()
    status_placeholder = st.empty()
    frame_placeholder = st.empty()

    for i in range(1, len(df_replay) + 1):
        df_slice = df_replay.iloc[:i]
        current_row = df_replay.iloc[i - 1]
        is_anomaly = current_row["timestamp"] in anomaly_timestamps

        # Update chart
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(df_slice["timestamp"], df_slice[replay_channel],
                color="steelblue", linewidth=1.5)

        anomaly_slice = df_slice[df_slice["timestamp"].isin(anomaly_timestamps)]
        if len(anomaly_slice) > 0:
            ax.scatter(anomaly_slice["timestamp"], anomaly_slice[replay_channel],
                      color="red", zorder=5, s=50)

        ax.set_xlim(df_replay["timestamp"].min(), df_replay["timestamp"].max())
        ax.set_ylim(df_replay[replay_channel].min() * 0.9,
                   df_replay[replay_channel].max() * 1.1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(replay_channel)
        ax.set_title(f"{replay_vehicle} — {replay_channel} replay")

        if is_anomaly:
            ax.axvline(x=current_row["timestamp"], color="red", 
                      linestyle="--", alpha=0.7)
            fig.patch.set_facecolor("#ffe6e6")

        chart_placeholder.pyplot(fig)
        plt.close()

        # Update map
        m_replay = folium.Map(
            location=[current_row["latitude"], current_row["longitude"]],
            zoom_start=16
        )

        # Draw route so far
        for _, row in df_slice.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=2,
                color="steelblue",
                fill=True,
                fill_opacity=0.5
            ).add_to(m_replay)

        # Current position
        folium.CircleMarker(
            location=[current_row["latitude"], current_row["longitude"]],
            radius=8,
            color="red" if is_anomaly else "green",
            fill=True,
            fill_opacity=1.0,
            popup="ANOMALY DETECTED!" if is_anomaly else "Normal"
        ).add_to(m_replay)

        if i % 10 == 0 or is_anomaly:
            map_placeholder.empty()
            with map_placeholder:
                st_folium(m_replay, width=1200, height=300, key=f"map_{i}")

        # Status
        frame_placeholder.caption(f"Frame {i}/{len(df_replay)} — Time: {current_row['timestamp']:.1f}s")

        if is_anomaly:
            status_placeholder.error(f"ANOMALY DETECTED at {current_row['timestamp']:.1f}s — {replay_channel}: {current_row[replay_channel]:.4f}")
        else:
            status_placeholder.success(f"Normal — {replay_channel}: {current_row[replay_channel]:.4f}")

        time.sleep(delay)

    st.success("Replay complete!")