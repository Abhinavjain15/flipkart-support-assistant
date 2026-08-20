"""
Tool: classify_product_image
Loads Part 2's saved transfer-learning classifier and predicts a category
label + confidence for a real .png file in data/sample_images/.
"""

import os
import sys
from typing import cast

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from part2_image_classifier.model import build_backbone, ClassifierHead, FullModel
from part2_image_classifier.preprocess import inference_transform
from part2_image_classifier.data_loader import CLASS_NAMES

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "product_classifier.pt"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None


def _load_model():
    global _model
    if _model is None:
        backbone = build_backbone().to(DEVICE)
        head = ClassifierHead().to(DEVICE)
        model = FullModel(backbone, head).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        _model = model
    return _model


def classify_product_image(image_path: str) -> dict:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = _load_model()
    image = Image.open(image_path).convert("L")
    tensor = cast(torch.Tensor, inference_transform(image)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(
            tensor
        )  # real call to Part 2's saved backbone+head, not a stand-in
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)

    predicted_index = int(pred_idx.item())
    return {
        "predicted_category": CLASS_NAMES[predicted_index],
        "confidence": round(confidence.item(), 4),
        "image_path": image_path,
    }


if __name__ == "__main__":
    sample_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "sample_images"
    )
    sample_files = sorted(os.listdir(sample_dir))[:3]

    for fname in sample_files:
        path = os.path.join(sample_dir, fname)
        result = classify_product_image(path)
        print(
            f"{fname:25s} -> predicted={result['predicted_category']:15s} confidence={result['confidence']}"
        )
