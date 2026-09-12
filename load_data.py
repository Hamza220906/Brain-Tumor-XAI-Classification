import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. ডেটাসেটের পাথ
# যেহেতু load_data.py এবং Training ফোল্ডার একই জায়গায় আছে, তাই './Training' লিখলেই চলবে।
data_dir = './Training' 

# চেক করা যাক ফোল্ডারটি আসলেই আছে কিনা
if not os.path.exists(data_dir):
    print(f"❌ Error: '{data_dir}' folder not found!")
    print("Please make sure the 'Training' folder is in the same directory as load_data.py")
else:
    print(f"✅ Found folder: {data_dir}")

    # 2. ইমেজ প্রি-প্রসেসিং
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        # 3. ডেটা লোড করা
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        
        # 4. ডেটা ব্যাচে ভাগ করা
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        print(f"🎉 Success! Total Images Loaded: {len(dataset)}")
        print(f"📂 Classes found: {dataset.classes}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")