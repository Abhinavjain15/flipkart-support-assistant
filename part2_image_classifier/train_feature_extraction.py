import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from model import ClassifierHead

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Documented hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 15
OPTIMIZER = "Adam"

cache = torch.load("cached_features.pt")
train_feats, train_labels = cache["train_feats"], cache["train_labels"]
val_feats, val_labels = cache["val_feats"], cache["val_labels"]

train_ds = TensorDataset(train_feats, train_labels)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

head = ClassifierHead().to(DEVICE)
optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

val_feats_dev = val_feats.to(DEVICE)
val_labels_np = val_labels.numpy()

best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    head.train()
    epoch_loss = 0.0
    for feats, labels in train_loader:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = head(feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * feats.size(0)
    epoch_loss /= len(train_ds)

    head.eval()
    with torch.no_grad():
        val_logits = head(val_feats_dev)
        val_preds = val_logits.argmax(dim=1).cpu().numpy()
    val_acc = accuracy_score(val_labels_np, val_preds)
    best_val_acc = max(best_val_acc, val_acc)

    print(f"Epoch {epoch:2d}/{EPOCHS}  loss={epoch_loss:.4f}  val_acc={val_acc:.4f}")

print(f"\nFeature-extraction phase best val accuracy: {best_val_acc:.4f}")
print(
    f"Config: batch_size={BATCH_SIZE}, optimizer={OPTIMIZER}, lr={LEARNING_RATE}, epochs={EPOCHS}"
)

torch.save(head.state_dict(), "head_feature_extraction.pt")
print("Saved: head_feature_extraction.pt")

if best_val_acc < 0.80:
    print("\n>>> Val accuracy below 80% -- fine-tuning (Task 4) is REQUIRED. <<<")
else:
    print(
        "\n>>> Val accuracy >= 80% -- feature extraction alone is sufficient, fine-tuning optional. <<<"
    )
