import matplotlib.pyplot as plt

def plot_column(df, column, output_path="output/plots/chart.png"):
    """Plot a single sensor column over time and save as an image."""
    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp"], df[column], label=column, color="steelblue")
    plt.xlabel("Time (s)")
    plt.ylabel(column)
    plt.title(f"Sensor Data: {column}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Chart saved to {output_path}") 
