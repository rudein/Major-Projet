Video Forgery Detection Using Deep Learning
This project is a deep learning-based solution for detecting forged videos. It processes video files frame by frame using a pre-trained Convolutional Neural Network (CNN) and determines if the video is real or forged based on the percentage of frames classified as fake.
Features
Detects forged videos with a threshold-based classification.
Processes videos frame by frame.
User-friendly GUI with a progress bar for real-time feedback.
Displays detailed results, including forged percentage and classification.

Tech Stack
Python 3.x
TensorFlow/Keras (Deep learning framework)
OpenCV (Video processing)
Tkinter (GUI)
NumPy (Numerical computations)

Dataset
This project uses the FaceForensics++ dataset, a benchmark for detecting manipulated videos. You can download it from the FaceForensics++ GitHub repository.

Setup
Prerequisites
Ensure the following:
Python 3.8 or higher is installed.
GPU support is optional but recommended.
Installation
Clone the repository:

 git clone https://github.com/rudein/Major-Projet.git
cd Major-Projet
Install the required dependencies: pip install -r requirements.tx
Place your trained model (forgery_detection_model.h5) in the root directory. You can use a pre-trained model or train one using the FaceForensics++ dataset.



Usage
Run the detection script:

 python detect_forgery.py


The GUI will open. Click "Open Video" to select a video file.


The progress bar will show the processing status. Once completed, a detailed result will be displayed:


Total frames analyzed.
Fake frames detected.
Forged percentage.
Final result (Real or Forged).

Configuration
Threshold
Videos with more than 40% forged frames are classified as Forged by default.
To adjust this threshold, modify the following line in detect_forgery.py:
 result = "Forged" if forged_percentage > 40 else "Real"

Training the Model
If you'd like to train your own model:
Extract frames from the FaceForensics++ dataset.
Organize the data into train and validation directories.
Use a CNN (e.g., MobileNet or ResNet50) with transfer learning.
Save the trained model as forgery_detection_model.h5.

Dependencies
All dependencies are listed in the requirements.txt file. Install them using:
pip install -r requirements.txt

Acknowledgments
FaceForensics++ Dataset
TensorFlow/Keras and OpenCV for providing the tools used in this project.


