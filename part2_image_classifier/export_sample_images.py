import os
import numpy as np
from PIL import Image
from torchvision import datasets
from data_loader import CLASS_NAMES

# Export real test-split images (raw IDX -> actual .png files).
raw_test = datasets.FashionMNIST(
    root="./data/fashion_mnist", train=False, download=True
)

OUT_DIR = "../data/sample_images"
os.makedirs(OUT_DIR, exist_ok=True)

# Pick one example per class from indices [0..N), covering 5+ distinct classes.
SAMPLE_CLASS_IDXS = [
    0,
    1,
    2,
    5,
    7,
    8,
    9,
]  # T-shirt/top, Trouser, Pullover, Sandal, Sneaker, Bag, Ankle boot
picked = {}
for idx in range(len(raw_test)):
    _, label = raw_test[idx]
    if label in SAMPLE_CLASS_IDXS and label not in picked:
        picked[label] = idx
    if len(picked) == len(SAMPLE_CLASS_IDXS):
        break

for i, (label, idx) in enumerate(sorted(picked.items())):
    img_array, _ = raw_test.data[idx].numpy(), raw_test.targets[idx].item()
    img = Image.fromarray(img_array)
    class_name = CLASS_NAMES[label].replace("/", "_").lower()
    filename = f"{i:02d}_{class_name}.png"
    img.save(os.path.join(OUT_DIR, filename))
    print(f"Saved: {filename}  (true label: {CLASS_NAMES[label]})")

print(f"\nTotal exported: {len(picked)} images to {OUT_DIR}/")
