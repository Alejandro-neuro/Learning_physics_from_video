import numpy as np
import os
import cv2

import pandas as pd
import matplotlib.pyplot as plt

# Function to iterate over folders and process video files
def iterate_and_process_videos(base_path):
    for root, dirs, files in os.walk(base_path):
        # If no subdirectories, we are in a terminal folder
        if not dirs:
            # Iterate over mp4 files that don't contain 'mask' in the filename
            mp4_files = [f for f in files if f.endswith('.mp4') and 'mask' not in f]
            for mp4_file in mp4_files:
                video_path = os.path.join(root, mp4_file)
                print(f"Processing video: {video_path}")
                if os.path.basename(root)=="":
                    numpy_save_path = os.path.join(root, f"{mp4_file.split('.')[0]}.npy")
                else:
                    numpy_save_path = os.path.join(root, f"{os.path.basename(root)}.npy")
                print(f"Saving numpy array to: {numpy_save_path}")
                video_array = process_video(video_path, new_width=100, new_height=56)
                np.save(numpy_save_path, video_array)

# Function to process video: resize, convert to grayscale, and normalize
def process_video(video_path, new_width, new_height):
    print('---------------------------------------------------------')    
    last_two_folders = video_path.split('/')[-3:-1]
    last_two_folders = '_'.join(last_two_folders)
    cap = cv2.VideoCapture(video_path)
    frames = []

    #get frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f'Original video fps: {fps}')
    

    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return np.array([])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        # Normalize the frame to range [0, 1]
        normalized_frame = gray_frame / 255.0
        frames.append(normalized_frame)
    
    cap.release()
    
    # Convert list of frames to numpy array  
    dataset = []
    np_frames = np.array(frames)
   
    print("number of frames:", np_frames.shape)
    nframes = np_frames.shape[0]
  

    step = 6 # frames to skip
    nf = 10 # number of frames per sample
    max_frames = nframes - (nf*step)
    for i in range(max_frames):

        end = i + nf*step
      

        frames_temp = np_frames[i:end:step]
        
        dataset.append(np.expand_dims( frames_temp, axis=1))
    dt = (step)/60
    dataset =  np.array(dataset)   

    print(f"Time between frames: {dt} ")
    print(f"Shape of dataset: {dataset.shape}")

    samples_number = -1
    sample_frames = dataset[samples_number,0,:,:]
    for i in range(1, dataset.shape[1]):
        sample_frames = np.hstack((sample_frames, dataset[samples_number,i,:,:]))
    
    sample_frames = sample_frames.transpose(1,2,0)
    plt.figure(figsize=(5,25))
    plt.imshow(sample_frames)
    
    if not os.path.exists('./Frames'):
        os.makedirs('./Frames')
    plt.savefig(f'./Frames/{last_two_folders}_sample_frame.png', dpi=300)
    plt.close()


    
    print('---------------------------------------------------------')

    return dataset

if __name__ == "__main__":

    base_path = "./Folder/torricelli/"  # Change to your base folder path
    iterate_and_process_videos(base_path)
