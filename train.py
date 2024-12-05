import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0
from sklearn.model_selection import train_test_split

# Define Video Dataset Class
class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame
            frame = cv2.resize(frame, (224, 224))  # Change to your model's input size
            frames.append(frame)
        cap.release()

        # Convert frames to tensor and stack them
        frames = torch.tensor(frames).permute(0, 3, 1, 2)  # Convert to (N, C, H, W)
        
        if self.transform:
            frames = self.transform(frames)

        return frames.float(), torch.tensor(label).long()  # Return as float tensor and long tensor

# Load dataset
def load_dataset(root_dir):
    video_paths = []
    labels = []
    
    for label, folder in enumerate(['real', 'forged']):
        folder_path = os.path.join(root_dir, folder)
        for video_file in os.listdir(folder_path):
            video_paths.append(os.path.join(folder_path, video_file))
            labels.append(label)

    return video_paths, labels

# Training Function
def train_model(model, criterion, optimizer, train_loader, num_epochs=5):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Main Function
def main():
    # Load dataset
    root_dir = "C:/Users/rudei/newmodel/data"
    video_paths, labels = load_dataset(root_dir)
    
    # Split into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(video_paths, labels, test_size=0.2, random_state=42)

    # Create Dataset and DataLoader
    train_dataset = VideoDataset(train_paths, train_labels)
    val_dataset = VideoDataset(val_paths, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Load model
    model = efficientnet_b0(num_classes=2)  # Assuming binary classification (real vs forged)
    model = model.cuda()  # Use GPU if available
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, criterion, optimizer, train_loader, num_epochs=10)

if __name__ == "__main__":
    main()
