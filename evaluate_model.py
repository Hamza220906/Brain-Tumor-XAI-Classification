import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Device Setup
device = torch.device('cpu')

# 2. Load Test Data (আমরা Training ডেটা থেকেই টেস্ট করছি কারণ আলাদা টেস্ট সেট নেই)
data_dir = './Training'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
class_names = dataset.classes

# 3. Load Trained Model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load('brain_tumor_model_test.pth', map_location=device))
model.eval()

# 4. Prediction Loop
all_preds = []
all_labels = []

print("🔍 Evaluating Model...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# 5. Generate Metrics
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# 6. Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
plt.show()
print("✅ Confusion Matrix saved as 'confusion_matrix.png'")