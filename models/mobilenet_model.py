from torchvision.models import mobilenet_v3_small


class MobileNetFER(nn.Module):
    """基于MobileNetV3的表情识别模型"""

    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = mobilenet_v3_small(pretrained=True)
        # 修改第一层以接受灰度图（3通道）
        self.backbone.features[0][0] = nn.Conv2d(3, 16, 3, 2, 1)
        # 替换分类头
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)