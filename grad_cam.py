import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image


device = torch.device('cpu') 
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('brain_tumor_model_test.pth', map_location=device))
model.eval()


target_layer = model.layer4[-1]


activations = {}
gradients = {}

def forward_hook(module, input, output):
    activations['value'] = output

def backward_hook(module, grad_in, grad_out):
    gradients['value'] = grad_out[0]

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0), img


def generate_grad_cam(image_path):
    input_tensor, original_img = preprocess_image(image_path)
    
    
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    
    model.zero_grad()
    output[0][predicted_class].backward()
    
    
    grads = gradients['value'].squeeze()
    acts = activations['value'].squeeze()
    weights = torch.mean(grads, dim=(1, 2))
    
    
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]
    
    cam = np.maximum(cam.detach().numpy(), 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    
    original_img_np = np.array(original_img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img_np, 0.6, heatmap, 0.4, 0)
    
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f'Original: {class_names[predicted_class]}')
    plt.imshow(original_img_np)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Grad-CAM Heatmap')
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.savefig('grad_cam_result.png')
    plt.show()
    print("✅ Grad-CAM image saved as 'grad_cam_result.png'")

from clinical_support import TREATMENT_GUIDELINES, DISCLAIMER

def generate_grad_cam(image_path):
    
    
    input_tensor, original_img = preprocess_image(image_path)
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    predicted_label = class_names[predicted_class]
    
    
   
    print(f"\n{'='*60}")
    print(f" DIAGNOSIS: {predicted_label.upper()}")
    print(f"⏱️ URGENCY: {TREATMENT_GUIDELINES[predicted_label]['urgency']}")
    print(f" PROTOCOL: {TREATMENT_GUIDELINES[predicted_label]['protocol']}")
    print(f"📅 FOLLOW-UP: {TREATMENT_GUIDELINES[predicted_label]['follow_up']}")
    print(f"\n{DISCLAIMER}")
    print(f"{'='*60}\n")
    
    
    plt.show()


sample_image_path = 'C:/Users/Hamza/Downloads/Compressed/BrainTumor/archive/Training/meningioma/Tr-me_50.jpg' 

try:
    generate_grad_cam(sample_image_path)
except Exception as e:
    print(f"Error: {e}")
    print("Please check if the image path is correct.")