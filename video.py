import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tensorflow.keras.models import load_model

# Load the Keras model
model = load_model(r"C:/Users/rudei/newmodel/video_forgery_model.h5")  # Update with your model path

# Function to preprocess the video frames
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to the input size expected by the model
        frame = cv2.resize(frame, (224, 224))  # Change to your model's input size
        frames.append(frame)
    cap.release()

    # Convert frames to numpy array and normalize
    frames = np.array(frames) / 255.0  # Normalize to [0, 1]
    
    return frames

# Function to make predictions
def predict_video(video_path, progress_bar):
    frames = preprocess_video(video_path)
    
    # Update progress bar
    progress_bar['maximum'] = len(frames)
    frame_predictions = []

    for i, frame in enumerate(frames):
        prediction = model.predict(np.expand_dims(frame, axis=0))  # Predict each frame
        frame_predictions.append(prediction[0][0])  # Store the prediction score
        progress_bar['value'] = i + 1  # Update progress bar
        root.update_idletasks()  # Update the GUI
    
    # Average predictions (or handle as necessary)
    average_prediction = np.mean(frame_predictions)

    # Check if average_prediction is less than or greater than 0.5
    if average_prediction > 0.6:  # Adjust threshold as needed
        messagebox.showinfo("Result", f"The video is forged. \nForgery score: {average_prediction:.2f}")
    else:
        messagebox.showinfo("Result", f"The video is real. \nAuthenticity score: {1 - average_prediction:.2f}")

# Function to open a file dialog and start prediction
def select_video():
    video_path = filedialog.askopenfilename(
        title="Select a Video",
        filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
    )
    if video_path:
        progress_bar['value'] = 0  # Reset progress bar
        predict_video(video_path, progress_bar)

# Set up the main window
root = tk.Tk()
root.title("Video Forgery Detection")

# Create a button to select a video
select_button = tk.Button(root, text="Select Video", command=select_video)
select_button.pack(pady=20)

# Create a progress bar
progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=20)

# Run the GUI
root.mainloop()
