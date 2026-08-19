import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

torch.manual_seed(42)
np.random.seed(42)

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_fashion_mnist(root="./data/fashion_mnist", val_size=5000):
    # Raw tensors only here; ImageNet normalization/resizing is applied later
    # by the backbone-specific preprocessing pipeline (Part 2 Task 2).
    base_transform = transforms.ToTensor()

    full_train = datasets.FashionMNIST(
        root=root, train=True, download=True, transform=base_transform
    )
    test_set = datasets.FashionMNIST(
        root=root, train=False, download=True, transform=base_transform
    )

    # Stratified train/val split on labels (train_test_split needs array-like labels)
    train_labels = np.array(full_train.targets)
    train_idx, val_idx = train_test_split(
        np.arange(len(full_train)),
        test_size=val_size,
        random_state=42,
        stratify=train_labels,
    )

    train_subset = Subset(full_train, train_idx)
    val_subset = Subset(full_train, val_idx)

    return train_subset, val_subset, test_set


if __name__ == "__main__":
    train_subset, val_subset, test_set = load_fashion_mnist()
    print(f"Train split: {len(train_subset)}")
    print(f"Val split:   {len(val_subset)}")
    print(f"Test split:  {len(test_set)}")

    # Confirm stratification held (val label distribution close to uniform ~10%)
    val_labels = np.array([train_subset.dataset.targets[i] for i in val_subset.indices])
    unique, counts = np.unique(val_labels, return_counts=True)
    print("\nVal split class distribution:")
    for cls_idx, count in zip(unique, counts):
        print(f"  {CLASS_NAMES[cls_idx]:15s}: {count}")
