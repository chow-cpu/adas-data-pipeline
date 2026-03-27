from ingest import load_csv
from preprocess import clean, normalize_column
from detect import detect_outliers
from visualize import plot_column

def run_pipeline(filepath):
    print("=== ADAS Data Pipeline ===\n")

    # Step 1: Load
    df = load_csv(filepath)

    # Step 2: Clean
    df = clean(df)

    # Step 3: Normalize speed
    df = normalize_column(df, "speed_mps")

    # Step 4: Detect anomalies
    outliers = detect_outliers(df, "speed_mps")
    print(f"\nFlagged rows:\n{outliers}\n")

    # Step 5: Visualize
    plot_column(df, "speed_mps", output_path="output/plots/speed.png")
    plot_column(df, "radar_distance_m", output_path="output/plots/radar.png")

    print("Pipeline complete. Check the output/ folder.")

if __name__ == "__main__":
    run_pipeline("data/sample_run.csv")
