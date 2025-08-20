import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import pyaudio
import threading

# --- 1. Load SER Model and Config ---
# This model architecture MUST match the one in train_ser.py
class SERModel(nn.Module):
    def __init__(self, num_classes=5):
        super(SERModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 27, 128) # Adjust if your spectrogram size is different
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ser_model = SERModel(num_classes=5).to(device)

try:
    ser_model.load_state_dict(torch.load('models/ser_model.pth', map_location=device))
    ser_model.eval()
    print("✅ Speech Emotion Recognition Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'models/ser_model.pth' not found.")
    ser_model = None

# We will use these 5 emotions as trained
idx_to_label = {0: 'happy', 1: 'sad', 2: 'angry', 3: 'neutral', 4: 'fearful'}

# --- 2. Real-time Audio Processing ---
class AudioProcessor:
    def __init__(self, app_state):
        self.app_state = app_state
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=1,
                                  rate=22050,
                                  input=True,
                                  frames_per_buffer=1024)
        self.audio_buffer = np.array([], dtype=np.float32)
        self.is_running = False

    def start_listening(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._listen_loop)
        self.thread.daemon = True
        self.thread.start()

    def _listen_loop(self):
        print("🎤 Audio listener started...")
        while self.is_running:
            data = self.stream.read(1024)
            self.audio_buffer = np.append(self.audio_buffer, np.frombuffer(data, dtype=np.float32))
            
            # Process every 2 seconds of audio
            if len(self.audio_buffer) >= 22050 * 2:
                self.process_audio()
                self.audio_buffer = np.array([], dtype=np.float32)

    def process_audio(self):
        if ser_model is None:
            return

        y = self.audio_buffer
        sr = 22050
        
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Pad or truncate to match model input size
        target_shape = (128, 216) # Based on 5s audio, may need adjustment
        log_mel_spectrogram = librosa.util.fix_length(log_mel_spectrogram, size=target_shape[1], axis=1)
        
        spectrogram_tensor = torch.from_numpy(np.expand_dims(log_mel_spectrogram, axis=0)).unsqueeze(0).to(device, dtype=torch.float)

        with torch.no_grad():
            outputs = ser_model(spectrogram_tensor)
            _, predicted = torch.max(outputs.data, 1)
            emotion = idx_to_label[predicted.item()]
            self.app_state['voice_emotion'] = emotion
            print(f"🎤 Voice Emotion Detected: {emotion}")