import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from modt import test_loader  # ✅ only if defined properly
from torchvision import models

# Step 1: Recreate the architecture
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # 3 classes: Real, Edited, AI-Generated

# Step 2: Load the saved weights
model.load_state_dict(torch.load("xpose_epoch_8.pth", map_location=torch.device("cpu")))

# Step 3: Set to eval mode
model.eval()


# # Load trained model
# model = torch.load("xpose_epoch_8.pth", map_location=torch.device("cpu"))
# model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate
def evaluate_model(model, test_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['AI-generated', 'Edited', 'Real']))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['AI-generated', 'Edited', 'Real'],
                yticklabels=['AI-generated', 'Edited', 'Real'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Run evaluation
evaluate_model(model, test_loader)
