import torch
import json


model_path = r"C:\Users\Hamza\Downloads\Compressed\BrainTumor\archive\brain_tumor_model_test.pth"

try:
    
    model_weights = torch.load(model_path, map_location='cpu')
    
    
    output_path = r"C:\Users\Hamza\Downloads\Compressed\BrainTumor\archive\model_weights.json"
    with open(output_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in model_weights.items()}, f)
        
    print(f"✅ Success! Weights saved to: {output_path}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check if the .pth file exists at the specified path.")