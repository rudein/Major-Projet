import os
import cv2
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

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
        frame = cv2.resize(frame, (224, 224))  # Adjust based on model input size
        frames.append(frame)
    cap.release()

    frames = np.array(frames) / 255.0  # Normalize to [0, 1]
    return frames

# Function to make predictions
def predict_video(video_path):
    frames = preprocess_video(video_path)
    predictions = model.predict(frames)
    average_prediction = np.mean(predictions, axis=0)
    return 1 if average_prediction > 0.51 else 0  # Return 1 for "forged" and 0 for "real"

# Evaluation Function
def evaluate_model(test_videos, true_labels):
    y_true = true_labels  # Ground truth labels (1 for forged, 0 for real)
    y_pred = []

    for video_path in test_videos:
        y_pred.append(predict_video(video_path))

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    
    # Check for AUC-ROC calculation possibility
    auc_roc = None
    if len(set(y_true)) > 1:  # Ensure both 0 and 1 are present in y_true
        auc_roc = roc_auc_score(y_true, y_pred)
    
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Print metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    if auc_roc is not None:
        print("AUC-ROC:", auc_roc)
    else:
        print("AUC-ROC: Not defined (only one class present in y_true)")
    print("Confusion Matrix:\n", conf_matrix)

    # Save Confusion Matrix as an Image
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Forged'], yticklabels=['Real', 'Forged'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig("confusion_matrix.png")  # Save the confusion matrix as an image
    plt.close()

    # Plot ROC Curve if AUC-ROC was calculated
    if auc_roc is not None:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label="AUC-ROC curve (area = {:.2f})".format(auc_roc))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="best")
        plt.savefig("roc_curve.png")  # Save the ROC curve as an image
        plt.close()

# Example usage
if __name__ == "__main__":
    test_videos = [
        r"C:/Users/rudei/newmodel/data/real/real_video1.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video2.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video3.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video4.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video5.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video6.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video7.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video8.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video9.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video10.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video11.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video12.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video13.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video14.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video15.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video16.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video17.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video18.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video19.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video20.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video21.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video22.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video23.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video24.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video25.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video26.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video27.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video28.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video29.mp4",
        r"C:/Users/rudei/newmodel/data/real/real_video30.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video1.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video2.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video3.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video4.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video5.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video6.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video7.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video8.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video9.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video10.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video11.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video12.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video13.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video14.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video15.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video16.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video17.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video18.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video19.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video20.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video21.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video22.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video23.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video24.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video25.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video26.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video27.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video28.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video29.mp4",
        r"C:/Users/rudei/newmodel/data/forged/forged_video30.mp4",  

        # Add remaining forged video paths here as in your list above
    ]
    
    true_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Update as needed

    # Run evaluation
    evaluate_model(test_videos, true_labels)
