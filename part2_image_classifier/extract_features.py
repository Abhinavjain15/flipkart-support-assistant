import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from data_loader import load_fashion_mnist
from model import build_backbone
from preprocess import inference_transform

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128


def apply_transform(subset, transform):
    """Wrap a Subset/dataset so each __getitem__ applies the ImageNet transform,
    overriding FashionMNIST's default (which was left as bare ToTensor in data_loader).
    """
    subset.dataset.transform = transform
    return subset


@torch.no_grad()
def extract_features(dataset, backbone, batch_size=BATCH_SIZE):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_features, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        feats = backbone(images)  # single frozen forward pass per image, cached once
        all_features.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)


if __name__ == "__main__":
    train_subset, val_subset, test_set = load_fashion_mnist()

    # FashionMNIST base transform was ToTensor only; swap to full ImageNet pipeline.
    train_subset.dataset.transform = inference_transform
    val_subset.dataset.transform = inference_transform
    test_set.transform = inference_transform

    backbone = build_backbone().to(DEVICE)

    print("Extracting train features (55000 images, one frozen forward pass)...")
    train_feats, train_labels = extract_features(train_subset, backbone)
    print(f"  train_feats: {train_feats.shape}")

    print("Extracting val features (5000 images)...")
    val_feats, val_labels = extract_features(val_subset, backbone)
    print(f"  val_feats: {val_feats.shape}")

    print("Extracting test features (10000 images)...")
    test_feats, test_labels = extract_features(test_set, backbone)
    print(f"  test_feats: {test_feats.shape}")

    torch.save(
        {
            "train_feats": train_feats,
            "train_labels": train_labels,
            "val_feats": val_feats,
            "val_labels": val_labels,
            "test_feats": test_feats,
            "test_labels": test_labels,
        },
        "cached_features.pt",
    )
    print("\nSaved: cached_features.pt")
