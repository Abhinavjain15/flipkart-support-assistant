import torch
from torchvision import transforms

try:
    from .model import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    from model import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

# Replicate grayscale -> 3ch, resize to backbone's expected input, ImageNet-normalize.
inference_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)
