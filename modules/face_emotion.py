import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# একটি সাধারণ CNN আর্কিটেকচার যা FER2013 ডেটাসেটের জন্য উপযুক্ত
# আপনার 'fer_model.pth' ফাইলটি অবশ্যই এই আর্কিটেকচারের সাথে মিলতে হবে
class FERModel(nn.Module):
    def __init__(self):
        super(FERModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 7)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = self.fc3(x)
        return x

# গ্লোবাল ভেরিয়েবল যেন মডেল একবারই লোড হয়
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FERModel().to(device)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# মডেল লোড করার চেষ্টা করা হচ্ছে
try:
    # আপনার প্রশিক্ষিত মডেলের পাথ দিন
    model.load_state_dict(torch.load('models/fer_model.pth', map_location=device))
    model.eval()
    print("✅ Facial Emotion Recognition Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'models/fer_model.pth' not found.")
    print("Please make sure the pre-trained model file exists in the 'models' directory.")
    model = None

# ছবি প্রি-প্রসেস করার জন্য ট্রান্সফর্ম
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict_emotion(face_image_np):
    """
    একটি মুখমণ্ডলের ছবি (numpy array) থেকে আবেগ প্রেডিক্ট করে।
    """
    if model is None:
        return "Model not loaded"
        
    image = Image.fromarray(face_image_np)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        emotion = emotion_labels[predicted_idx.item()]
        
    return emotion