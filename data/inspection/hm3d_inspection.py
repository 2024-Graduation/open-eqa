import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt

def import_csv():
    with open('./data/statistics/video_statistics.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)

        # Delete a header
        next(csv_reader)

        # Return a numpy array
        return np.array([row for row in csv_reader])
    
def plot_video_statistics(csv):
    os.makedirs(os.path.join(os.getcwd(), "data", "results"), exist_ok=True)
    
    plot_video_length_histogram(csv)
    plot_number_of_frames_histogram(csv)
    plot_visited_rooms_bar(csv)
    plot_moved_floors_bar(csv)

def plot_video_length_histogram(csv, filename='video_length_histogram.png'):
    # Plot the histogram that shows a length of each video
    plt.figure()
    data = csv[:, 1].astype(int)
    bins = np.arange(data.min(), data.max() + 5, 5)
    counts, bins, _ = plt.hist(data, bins=bins, edgecolor='black')
    plt.xlabel('Video Length')
    plt.ylabel('Frequency')

    # Add counts on top of the bars
    for count, bin_edge in zip(counts, bins[:-1]):
        if count > 0:
            plt.text(bin_edge + 2.5, count, str(int(count)), ha='center', va='bottom')

    # Set y-axis to integer range
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.plot()

    # Save the histogram
    plt.savefig(os.path.join(os.getcwd(), "data", "results", filename))
    plt.close()

def plot_number_of_frames_histogram(csv, filename='number_of_frames_histogram.png'):
    # Plot the histogram that shows a number of frames in each video
    plt.figure()
    data = csv[:, 2].astype(int)
    bins = np.arange(data.min(), data.max() + 25, 25)
    counts, bins, _ = plt.hist(data, bins=bins, edgecolor='black')
    plt.xlabel('Number of Frames')
    plt.ylabel('Frequency')

    # Add counts on top of the bars
    for count, bin_edge in zip(counts, bins[:-1]):
        if count > 0:
            plt.text(bin_edge + 12.5, count, str(int(count)), ha='center', va='bottom')

    # Set y-axis to integer range
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.plot()

    # Save the histogram
    plt.savefig(os.path.join(os.getcwd(), "data", "results", filename))
    plt.close()

def plot_visited_rooms_bar(csv, filename='visited_rooms_bar.png'):
    # Plot the bar graph that shows a number of visited rooms in each video
    plt.figure()
    data = csv[:, 3].astype(int)
    unique, counts = np.unique(data, return_counts=True)
    plt.bar(unique, counts, edgecolor='black', width=0.8)
    plt.xlabel('Visited Rooms')
    plt.ylabel('Frequency')

    # Add counts on top of the bars
    for i in range(len(unique)):
        plt.text(unique[i], counts[i], str(counts[i]), ha='center', va='bottom')

    # Set y-axis to integer range
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.plot()

    # Save the bar graph
    plt.savefig(os.path.join(os.getcwd(), "data", "results", filename))
    plt.close()

def plot_moved_floors_bar(csv, filename='moved_floors_bar.png'):
    # Plot the bar graph that shows a number of moved floors in each video
    plt.figure()
    data = csv[:, 4].astype(int)
    unique, counts = np.unique(data, return_counts=True)
    plt.bar(unique, counts, edgecolor='black', width=0.4)
    plt.xlabel('Moved Floors')
    plt.ylabel('Frequency')

    # Add counts on top of the bars
    for i in range(len(unique)):
        plt.text(unique[i], counts[i], str(counts[i]), ha='center', va='bottom')

    # Set y-axis to integer range
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Set x-axis to integer range
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.plot()

    # Save the bar graph
    plt.savefig(os.path.join(os.getcwd(), "data", "results", filename))
    plt.close()

def main():
    csv = import_csv()
    plot_video_statistics(csv)

if __name__ == "__main__":
    main()