import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import os
model_path = "xpose_epoch_8.pth"
image_path = "fake_558.jpg"
# Load class labels
class_names = ['ai_generated', 'edited', 'real']  # Adjust based on your folder names

# Define image transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # Mean
                         [0.229, 0.224, 0.225])   # Std
])

# Load model
def load_model(model_path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Predict function
def predict(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probs[0][predicted].item()

    predicted_class = class_names[predicted.item()]
    print(f"Prediction: {predicted_class} (Confidence: {confidence*100:.2f}%)")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--model", default="xpose_epoch_5.pth", help="Path to model checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("❌ Image path is invalid.")
        exit(1)
    if not os.path.exists(args.model):
        print("❌ Model checkpoint not found.")
        exit(1)

    model = load_model(args.model)
    predict(args.image, model)
