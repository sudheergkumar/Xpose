import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import os
print("🚀 Starting training script...")

# Configuration
DATA_DIR = "model"
BATCH_SIZE = 8
EPOCHS = 5
NUM_CLASSES = 3  # real, edited, ai_generated
MODEL_PATH = "image_classifier.pth"

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print("📂 Data loaded successfully.")

# Load pre-trained model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}")
print("✅ Training finished. Saving model now...")

# Save the model
torch.save(model, MODEL_PATH)
torch.save(model, "image_classifier.pth")
print("💾 Model saved as image_classifier.pth")

print(f"✅ Model saved as {MODEL_PATH}")
