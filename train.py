import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import argparse

from models.cnn_model import EnhancedCNN
from utils.dataset import get_data_loaders
from utils.visualization import plot_training_curves


def train_model(model, train_loader, val_loader, num_classes=7, epochs=100, device='cuda'):
    """训练模型"""

    # 使用类别权重（处理不平衡问题）
    class_weights = torch.tensor([1.0, 1.5, 1.2, 0.8, 1.0, 1.0, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc = 0
    patience_counter = 0
    early_stop_patience = 15

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / len(train_loader.dataset)
        train_loss_avg = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / len(val_loader.dataset)
        val_loss_avg = val_loss / len(val_loader)

        # 记录
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss_avg)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': train_loader.dataset.classes  # 保存类别名称
            }, 'best_model.pth')
            patience_counter = 0
            print(f"✓ 保存最佳模型，验证准确率: {val_acc:.2f}%")
        else:
            patience_counter += 1

        print(f'Epoch [{epoch + 1:3d}/{epochs}] '
              f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')

        if patience_counter >= early_stop_patience:
            print(f"\n早停触发，最佳验证准确率: {best_val_acc:.2f}%")
            break

    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data',
                        help='数据集根目录（应包含 train/ 和 val/ 子文件夹）')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print("加载数据...")
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        args.data_dir, args.batch_size
    )

    # 创建模型
    model = EnhancedCNN(num_classes=len(class_names)).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    print("\n开始训练...")
    best_acc = train_model(model, train_loader, val_loader,
                           len(class_names), args.epochs, device)
    print(f"\n训练完成！最佳验证准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()