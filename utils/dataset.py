import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


def get_data_loaders(data_dir, batch_size=64, num_workers=4):
    """
    从文件夹结构加载图片数据集

    Args:
        data_dir: 数据集根目录，应包含 train/ 和 val/ 两个子文件夹
        batch_size: 批次大小
        num_workers: 数据加载线程数

    Returns:
        train_loader, val_loader, test_loader
    """

    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 转为3通道（适配模型）
        transforms.RandomRotation(15),  # 随机旋转
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 亮度对比度调整
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        transforms.Resize((48, 48)),  # 调整到48x48
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 验证集/测试集：仅缩放和归一化，不做数据增强
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 使用 ImageFolder 直接从文件夹读取图片
    # ImageFolder 会自动将子文件夹名作为类别标签[citation:7]
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transform
    )

    # 如果有测试集文件夹，也加载
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(
            root=test_dir,
            transform=val_transform
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
    else:
        test_loader = None

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    # 打印类别信息
    print(f"发现 {len(train_dataset.classes)} 个表情类别:")
    for idx, class_name in enumerate(train_dataset.classes):
        print(f"  {idx}: {class_name}")

    print(f"\n训练集图片数: {len(train_dataset)}")
    print(f"验证集图片数: {len(val_dataset)}")

    return train_loader, val_loader, test_loader, train_dataset.classes