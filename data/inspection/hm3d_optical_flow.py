import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_optical_flow(frame1, frame2):
    """
    Calculate the optical flow between two frames using Gunner Farneback's algorithm.
    
    Args:
    - frame1: First frame (as a numpy array).
    - frame2: Second frame (as a numpy array).
    
    Returns:
    - flow: Optical flow (as a numpy array).
    """
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(gray_frame1, gray_frame2, None, 
                                        pyr_scale=0.5, levels=3, winsize=15, 
                                        iterations=3, poly_n=5, poly_sigma=1.2, 
                                        flags=0)
    return flow

def calculate_dynamics(flow):
    """
    Calculate the dynamics (mean magnitude and variance of magnitude) of the optical flow.
    
    Args:
    - flow: Optical flow (as a numpy array).
    
    Returns:
    - mean_magnitude: Mean magnitude of the optical flow.
    - variance_magnitude: Variance of the magnitude of the optical flow.
    """
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_magnitude = np.mean(magnitude)
    variance_magnitude = np.var(magnitude)
    return mean_magnitude, variance_magnitude

def process_videos_in_folder(folder_path, output_csv_path):
    """
    Process all videos in the given folder, calculate optical flow for each frame pair,
    and save the results to a CSV file.
    
    Args:
    - folder_path: Path to the folder containing video subfolders.
    - output_csv_path: Path to the output CSV file.
    """
    results = []
    
    for j, video_folder in enumerate(os.listdir(folder_path)):
        video_path = os.path.join(folder_path, video_folder)
        if not os.path.isdir(video_path):
            continue
        
        frames = sorted(os.listdir(video_path))
        if len(frames) < 2:
            continue
        
        video_mean_magnitudes = []
        video_variance_magnitudes = []
        
        for i in tqdm(range(len(frames) - 1), leave=False):
            frame1_path = os.path.join(video_path, frames[i])
            frame2_path = os.path.join(video_path, frames[i + 1])
            
            frame1 = cv2.imread(frame1_path)
            frame2 = cv2.imread(frame2_path)
            
            if frame1 is None or frame2 is None:
                continue
            
            flow = calculate_optical_flow(frame1, frame2)
            mean_magnitude, variance_magnitude = calculate_dynamics(flow)
            video_mean_magnitudes.append(mean_magnitude)
            video_variance_magnitudes.append(variance_magnitude)
        
        if video_mean_magnitudes and video_variance_magnitudes:
            avg_mean_magnitude = np.mean(video_mean_magnitudes)
            avg_variance_magnitude = np.mean(video_variance_magnitudes)
            
            results.append({
                'video_folder': video_folder,
                'avg_mean_magnitude': avg_mean_magnitude,
                'avg_variance_magnitude': avg_variance_magnitude
            })

            print(f'#{j} - Processed video {video_folder}: avg_mean_magnitude={avg_mean_magnitude}, avg_variance_magnitude={avg_variance_magnitude}')
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f'Saved results to {output_csv_path}')

# Example usage
if __name__ == "__main__":
    folder_path = 'data/frames/hm3d-v0'
    output_csv_path = 'data/inspection/statistics/video_dynamics_results.csv'
    process_videos_in_folder(folder_path, output_csv_path)
