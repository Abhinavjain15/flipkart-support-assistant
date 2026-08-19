import torch
import torch.nn as nn
from torchvision import models

torch.manual_seed(42)

# ResNet-18 expects 224x224, 3-channel, ImageNet-normalized input.
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 10
FEATURE_DIM = 512  # ResNet-18's final conv output dim before its original fc layer


def build_backbone():
    """Pretrained ResNet-18 with its classification head stripped off.
    Early/middle conv layers are frozen; only later training (Task 4) may
    selectively unfreeze late layers if feature-extraction accuracy is low."""
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in backbone.parameters():
        param.requires_grad = False  # freeze everything for feature extraction
    backbone.fc = (
        nn.Identity()
    )  # expose 512-d pooled features instead of 1000-class logits
    backbone.eval()
    return backbone


class ClassifierHead(nn.Module):
    """New head trained from scratch on cached 512-d backbone features."""

    def __init__(self, in_dim=FEATURE_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FullModel(nn.Module):
    """Backbone + head combined, used for final saved inference (Task 7) and
    for optional fine-tuning (Task 4) where backbone params become trainable again."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
