import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# --- 1. Configuration and Constants ---
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = 'datasets/ravdess-audio' # RAVDESS ডেটাসেটের পাথ
SAMPLE_RATE = 22050
FIXED_LENGTH = 5 # সেকেন্ডে, অডিওর দৈর্ঘ্য নির্দিষ্ট করার জন্য

# RAVDESS ফাইলের নাম থেকে আবেগ শনাক্ত করার জন্য ম্যাপিং
emotion_map = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}
# আমরা যে আবেগগুলো ব্যবহার করব
selected_emotions = ["happy", "sad", "angry", "neutral", "fearful"]
label_to_idx = {emotion: i for i, emotion in enumerate(selected_emotions)}
idx_to_label = {i: emotion for emotion, i in label_to_idx.items()}

# --- 2. SER Model Architecture (CNN) ---
class SERModel(nn.Module):
    def __init__(self, num_classes):
        super(SERModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # The input features to the fully connected layer might need adjustment
        # based on the final size of the spectrogram after convolutions and pooling.
        # Let's calculate it: H_out = floor((H_in + 2*padding - kernel)/stride + 1)
        # Initial size: 128 (mels) x 216 (time steps for 5s audio)
        # After 3 pooling layers: H = 128/8=16, W = 216/8=27
        self.fc1 = nn.Linear(64 * 16 * 27, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return x

# --- 3. Custom Dataset Class ---
class RavdessDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Audio loading and padding/truncating
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        target_length = FIXED_LENGTH * sr
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, target_length - len(y)), 'constant')

        # Mel Spectrogram creation
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Add channel dimension (C, H, W)
        log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=0)

        if self.transform:
            log_mel_spectrogram = self.transform(log_mel_spectrogram)
            
        return log_mel_spectrogram, label

# --- Main execution block ---
if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    # 1. Prepare Data
    file_paths = []
    labels = []
    print("Scanning audio files...")
    for dirpath, _, filenames in os.walk(DATASET_PATH):
        for filename in filenames:
            if filename.endswith(".wav"):
                parts = filename.split("-")
                # ✅ Safely check if the filename has enough parts to prevent IndexError
                if len(parts) > 2:
                    emotion_code = parts[2]
                    if emotion_code in emotion_map and emotion_map[emotion_code] in selected_emotions:
                        file_paths.append(os.path.join(dirpath, filename))
                        labels.append(label_to_idx[emotion_map[emotion_code]])
    
    print(f"Found {len(file_paths)} relevant audio files.")
    
    # Train-test split
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 2. Create Datasets and DataLoaders
    train_dataset = RavdessDataset(train_paths, train_labels, transform=torch.from_numpy)
    test_dataset = RavdessDataset(test_paths, test_labels, transform=torch.from_numpy)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # Set num_workers=0 for Windows debugging
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # Set num_workers=0 for Windows debugging
    
    # 3. Initialize Model, Loss Function, and Optimizer
    num_classes = len(selected_emotions)
    model = SERModel(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training and Validation Loop
    best_val_accuracy = 0.0
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        model.train()
        progress_bar = tqdm(train_loader, desc="Training")
        for inputs, labels in progress_bar:
            inputs = inputs.to(DEVICE, dtype=torch.float)
            labels = labels.to(DEVICE, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Validation")
            for inputs, labels in progress_bar:
                inputs = inputs.to(DEVICE, dtype=torch.float)
                labels = labels.to(DEVICE, dtype=torch.long)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_acc = accuracy_score(all_labels, all_preds) * 100
        print(f"Validation Accuracy: {epoch_val_acc:.2f}%")
        
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            torch.save(model.state_dict(), 'models/ser_model.pth')
            print(f"✅ New best SER model saved with accuracy: {best_val_accuracy:.2f}%")

    print("\n--- SER Training Finished ---")