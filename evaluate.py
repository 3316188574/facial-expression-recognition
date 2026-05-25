# 快速评估脚本（使用验证集）
import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.cnn_model import EnhancedCNN

# 设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = 'data'  # 你的数据目录
model_path = 'best_model.pth'

# 加载数据
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 使用验证集
val_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, 'val'),
    transform=transform
)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

# 加载模型
model = EnhancedCNN(num_classes=len(val_dataset.classes)).to(device)
checkpoint = torch.load(model_path, map_location=device)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"最佳验证准确率: {checkpoint.get('val_acc', 'N/A')}%")
else:
    model.load_state_dict(checkpoint)

# 评估
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

print(f"\n验证集准确率: {100.*correct/total:.2f}%")