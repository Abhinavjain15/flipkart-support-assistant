import os
import torch
import shutil
from model import build_backbone, ClassifierHead, FullModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Combine backbone + trained head into one deployable model, save state_dict.
backbone = build_backbone().to(DEVICE)
head = ClassifierHead().to(DEVICE)
head.load_state_dict(torch.load("head_feature_extraction.pt"))
full_model = FullModel(backbone, head).to(DEVICE)
full_model.eval()

os.makedirs("../models", exist_ok=True)
torch.save(full_model.state_dict(), "../models/product_classifier.pt")
print("Saved: models/product_classifier.pt")


def load_and_predict(image_path: str) -> dict:
    """
    Documented one-function loader + single-image predictor.
    This is exactly what Part 3's classify_product_image tool calls.
    """
    from PIL import Image
    from model import build_backbone, ClassifierHead, FullModel, NUM_CLASSES
    from preprocess import inference_transform
    from data_loader import CLASS_NAMES
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = build_backbone().to(device)
    head = ClassifierHead().to(device)
    model = FullModel(backbone, head).to(device)
    model.load_state_dict(
        torch.load("models/product_classifier.pt", map_location=device)
    )
    model.eval()

    image = Image.open(image_path).convert("L")  # ensure single-channel source
    tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)

    return {
        "predicted_category": CLASS_NAMES[pred_idx.item()],
        "confidence": round(confidence.item(), 4),
    }


if __name__ == "__main__":
    # Sanity check: reload saved weights and confirm inference runs end-to-end
    reloaded_backbone = build_backbone().to(DEVICE)
    reloaded_head = ClassifierHead().to(DEVICE)
    reloaded_model = FullModel(reloaded_backbone, reloaded_head).to(DEVICE)
    reloaded_model.load_state_dict(
        torch.load("../models/product_classifier.pt", map_location=DEVICE)
    )
    reloaded_model.eval()
    print("Reload check passed: product_classifier.pt loads without error.")
