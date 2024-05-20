import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt

METADATA_PATH = 'data/inspection/statistics/metadata.csv'
NEW_METADATA_PATH = 'data/inspection/statistics/new_metadata.csv'

def plot_histogram(path, file_name):
    # Read the CSV files
    metadata = pd.read_csv(path)

    # Columns to plot histograms for
    columns = ['num_floors', 'floor_space', 'num_rooms']

     # Columns to plot histograms for
    columns = ['num_floors', 'floor_space', 'num_rooms']
    x_labels = ['# floors', 'floor area (m^2)', '# rooms']
    x_ranges = [(0, 8), (0, 1500), (0, 35)]
    colors = ['C0', 'C1', 'C3']
    bins = [3, 15, 10] if "new" in file_name else [9, 100, 50]

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot histograms for each column
    for ax, col, x_label, x_range, color, bin in zip(axes, columns, x_labels, x_ranges, colors, bins):
        ax.hist(metadata[col], bins=bin, alpha=1.0, color=color, edgecolor='black', linewidth=0.7)
        ax.set_xlabel(x_label)
        ax.set_ylabel("# scenes")
        ax.set_xlim(x_range)  # Set the x-axis range

    # Save the histogram
    output_path = os.path.join(os.getcwd(), "data", "results", file_name)
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    plot_histogram(METADATA_PATH, "metadata_histogram.png")
    plot_histogram(NEW_METADATA_PATH, "new_metadata_histogram.png")