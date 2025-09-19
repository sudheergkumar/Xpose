import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset path
DATASET_DIR = "dataset"

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from PIL import Image
Image.MAX_IMAGE_PIXELS = None


# Load dataset
dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)

# Split dataset
total = len(dataset)
train_size = int(0.7 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

# Dataloaders
BATCH_SIZE = 32
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)

# Weighted loss
class_counts = torch.tensor([33000, 20000, 500], dtype=torch.float)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0005)

from PIL import Image
import os

def fix_large_images(root_folder, max_pixels=178956970):
    count = 0
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                path = os.path.join(root, file)
                try:
                    with Image.open(path) as img:
                        if img.width * img.height > max_pixels:
                            print(f"Resizing {file} ({img.width}x{img.height})...")
                            img = img.resize((1024, 1024))  # Resize to manageable size
                            img.save(path)
                            count += 1
                except Exception as e:
                    print(f"Could not process {path}: {e}")
    print(f"✅ Fixed {count} large images.")

# Run this before training
fix_large_images("dataset")


# Training function
import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss
        train_acc = correct / total
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # ✅ Save model after each epoch
        model_path = f"xpose_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved model to: {model_path}")

# Train the model
train_model(model, train_loader, val_loader, epochs=10)
