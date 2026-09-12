import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Brain Tumor XAI", layout="wide")
st.title("🧠 Brain Tumor Diagnosis & Explainable AI System")
st.markdown("*ResNet18 Classification + Grad-CAM Visualization + NCCN Guidelines*")

# Treatment Guidelines
TREATMENT_GUIDELINES = {
    'glioma': {'urgency': '🔴 CRITICAL', 'protocol': 'Surgical resection + Radiotherapy & Temozolomide', 'follow_up': 'MRI every 3 months'},
    'meningioma': {'urgency': '🟡 MODERATE', 'protocol': 'Observation or Surgical excision', 'follow_up': 'Annual MRI for 5 years'},
    'pituitary': {'urgency': '🟢 STANDARD', 'protocol': 'Transsphenoidal surgery or Dopamine agonists', 'follow_up': 'Hormone panel test in 6 months'},
    'notumor': {'urgency': '🔵 ROUTINE', 'protocol': 'No intervention required', 'follow_up': 'Repeat MRI in 6-12 months if symptoms persist'}
}
DISCLAIMER = "⚠️ AI-generated suggestion based on NCCN guidelines. Final decision must be made by a qualified neuro-oncologist."

# Model & Grad-CAM Setup
@st.cache_resource
def load_model_and_hooks():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    
    # Hugging Face 
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id="Hamza220906/brain-tumor-resnet18", 
        filename="brain_tumor_model_test.pth"
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    
    activations = {}
    gradients = {}
    def forward_hook(module, input, output):
        activations['value'] = output
    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]
    target_layer = model.layer4[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)
    
    return model, activations, gradients

model, activations, gradients = load_model_and_hooks()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload MRI Scan (JPG/PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    original_img = Image.open(uploaded_file).convert('RGB')
    input_tensor = transform(original_img).unsqueeze(0)
    
    # Forward Pass
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    diagnosis = class_names[pred_class]
    
    # Backward Pass for Grad-CAM
    model.zero_grad()
    score = output[0][pred_class]
    score.backward()
    
    # Generate Heatmap
    grads = gradients['value'].squeeze()
    acts = activations['value'].squeeze()
    weights = torch.mean(grads, dim=(1, 2))
    
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]
        
    cam = np.maximum(cam.detach().numpy(), 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    
    # Overlay
    img_np = np.array(original_img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption="Original MRI", use_container_width=True)
        st.image(overlay_rgb, caption="Grad-CAM Heatmap Overlay", use_container_width=True)
        
    with col2:
        st.success(f"Diagnosis: {diagnosis.upper()}")
        guideline = TREATMENT_GUIDELINES[diagnosis]
        st.warning(f"Urgency: {guideline['urgency']}")
        st.info(f"Protocol: {guideline['protocol']}")
        st.caption(f"Follow-up: {guideline['follow_up']}")
        st.error(DISCLAIMER)

st.divider()
st.caption("Developed by Hamza | Code: github.com/Hamza220906/Brain-Tumor-XAI-Classification")