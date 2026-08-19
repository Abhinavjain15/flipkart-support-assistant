import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from model import ClassifierHead
from data_loader import CLASS_NAMES

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cache = torch.load("cached_features.pt")
test_feats, test_labels = cache["test_feats"], cache["test_labels"]

head = ClassifierHead().to(DEVICE)
head.load_state_dict(torch.load("head_feature_extraction.pt"))
head.eval()

with torch.no_grad():
    logits = head(test_feats.to(DEVICE))
    preds = logits.argmax(dim=1).cpu().numpy()

y_true = test_labels.numpy()
test_acc = accuracy_score(y_true, preds)
print(f"Test accuracy: {test_acc:.4f}\n")

cm = confusion_matrix(y_true, preds)
print("=== Confusion matrix (rows=true, cols=predicted) ===")
header = "        " + " ".join(f"{i:5d}" for i in range(10))
print(header)
for i, row in enumerate(cm):
    print(f"{i:2d} {CLASS_NAMES[i]:12s}" + " ".join(f"{v:5d}" for v in row))

precision, recall, f1, support = precision_recall_fscore_support(
    y_true, preds, zero_division=0
)
print("\n=== Per-class precision/recall ===")
for i, name in enumerate(CLASS_NAMES):
    print(
        f"{name:15s} precision={precision[i]:.4f}  recall={recall[i]:.4f}  support={support[i]}"
    )

# Find top confused off-diagonal pairs (symmetric: i->j + j->i)
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
pair_counts = {}
for i in range(10):
    for j in range(10):
        if i != j:
            key = tuple(sorted((i, j)))
            pair_counts[key] = pair_counts.get(key, 0) + cm_no_diag[i, j]

top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:2]
print("\n=== Top confused category pairs ===")
for (i, j), count in top_pairs:
    print(f"{CLASS_NAMES[i]} <-> {CLASS_NAMES[j]}: {count} total misclassifications")

np.save("confusion_matrix.npy", cm)
print("\nSaved: confusion_matrix.npy")
print(
    f"\nTop pairs for writeup reference: {[(CLASS_NAMES[i], CLASS_NAMES[j]) for (i,j),_ in top_pairs]}"
)
