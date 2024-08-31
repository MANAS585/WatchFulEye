import cv2
import numpy as np
import os

# Function to calculate optical flow
def calculate_optical_flow(prev_frame, current_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate magnitude and angle of flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    return magnitude

# Function to detect and save Canny edge images of frames with aggressive motion in a video
def detect_and_save_aggressive_canny_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Open video capture
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, prev_frame = cap.read()
    frame_count = 0

    while True:
        # Read the current frame
        ret, current_frame = cap.read()
        if not ret:
            break

        # Calculate optical flow magnitude
        magnitude = calculate_optical_flow(prev_frame, current_frame)

        # Threshold for detecting aggressive motion
        threshold = 7

        # Apply Gaussian blur to reduce noise and improve sensitivity
        magnitude_blurred = cv2.GaussianBlur(magnitude, (5, 5), 0)

        # Count the number of pixels with magnitude above the threshold
        aggressive_pixels = np.sum(magnitude_blurred > threshold)

        # If a significant number of pixels have high magnitude, consider it aggressive motion
        if aggressive_pixels > 0.01 * magnitude.size:
            print(f"Aggressive motion detected in frame {frame_count}!")

            # Apply Canny edge detection to the frame
            edges = cv2.Canny(current_frame, 50, 150)

            # Save the Canny edge image
            output_filename = f"{video_name}_{frame_count:03d}_canny.png"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, edges)

        # Update the previous frame and frame count
        prev_frame = current_frame
        frame_count += 1

    # Release video capture
    cap.release()

# Function to process all videos in a given folder
def process_videos_in_folder(folder_path, output_root_folder):
    # Iterate through all folders in the specified directory
    for root, dirs, files in os.walk(folder_path):
        for folder in dirs:
            folder_path = os.path.join(root, folder)

            # Create the corresponding output folder in the root output folder
            output_folder = os.path.join(output_root_folder, folder)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Iterate through all video files in the current folder
            for file in os.listdir(folder_path):
                if file.endswith(".mp4"):
                    video_path = os.path.join(folder_path, file)
                    detect_and_save_aggressive_canny_frames(video_path, output_folder)

process_videos_in_folder('/Users/harshi/Desktop/Testing', '/Users/harshi/Desktop/Output_testing')
