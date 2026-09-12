import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


data_dir = './Training'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
class_names = dataset.classes
print(f"Classes: {class_names}")


model = models.resnet18(pretrained=False) 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, loader, epochs=1):
    model.train()
    losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        count = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            count += 1
            if count % 10 == 0: 
                print(f"  Processed {count * 64} images...")
        
        avg_loss = running_loss / len(loader)
        losses.append(avg_loss)
        print(f"✅ Epoch [{epoch+1}/{epochs}] Completed! Loss: {avg_loss:.4f}")
    
    return losses


print("🚀 Starting Quick Test Training (1 Epoch)...")
losses = train_model(model, train_loader, epochs=1)


torch.save(model.state_dict(), 'brain_tumor_model_test.pth')
print("✅ Test Model Saved!")