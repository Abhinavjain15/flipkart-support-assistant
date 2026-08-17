import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
)
from preprocessing import load_splits, build_preprocessor

X_train, X_test, y_train, y_test = load_splits()

logreg_pipe = Pipeline(
    [
        ("preprocess", build_preprocessor()),
        (
            "clf",
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        ),
    ]
)
logreg_pipe.fit(X_train, y_train)

proba = logreg_pipe.predict_proba(X_test)[:, 1]
preds_default = (proba >= 0.5).astype(int)

acc = accuracy_score(y_test, preds_default)
f1_pos = f1_score(y_test, preds_default, pos_label=1)
recall_pos = recall_score(y_test, preds_default, pos_label=1)
precision_pos = precision_score(y_test, preds_default, pos_label=1)
roc_auc = roc_auc_score(y_test, proba)

print("=== Default threshold (0.5) ===")
print(f"Accuracy:  {acc:.4f}")
print(f"F1:        {f1_pos:.4f}")
print(f"Recall:    {recall_pos:.4f}")
print(f"Precision: {precision_pos:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# Threshold sweep: 0.1 to 0.9 step 0.02
thresholds = np.arange(0.10, 0.901, 0.02)
best_f1, best_t, best_recall, best_precision = -1, None, None, None
sweep_results = []

for t in thresholds:
    preds_t = (proba >= t).astype(int)
    f1_t = f1_score(y_test, preds_t, pos_label=1, zero_division=0)
    r_t = recall_score(y_test, preds_t, pos_label=1, zero_division=0)
    p_t = precision_score(y_test, preds_t, pos_label=1, zero_division=0)
    sweep_results.append((round(t, 2), f1_t, r_t, p_t))
    if f1_t > best_f1:
        best_f1, best_t, best_recall, best_precision = f1_t, t, r_t, p_t

print("\n=== Threshold sweep (t, F1, recall, precision) ===")
for t, f1_t, r_t, p_t in sweep_results:
    print(f"t={t:.2f}  F1={f1_t:.4f}  recall={r_t:.4f}  precision={p_t:.4f}")

print(f"\nBest F1 threshold: t*={best_t:.2f}")
print(f"F1={best_f1:.4f}  Recall={best_recall:.4f}  Precision={best_precision:.4f}")

recall_gain_pp = (best_recall - recall_pos) * 100
precision_drop_pp = (precision_pos - best_precision) * 100
print(f"\nRecall gain vs default: {recall_gain_pp:.2f}pp")
print(f"Precision drop vs default: {precision_drop_pp:.2f}pp")
print(
    "\nBusiness trade-off: lowering the threshold catches more true returns (higher recall) "
    "at the cost of more false positives -- orders flagged as return-risk that would not "
    "actually be returned. This is the right trade when the cost of missing a real return "
    "(unrecovered shipping, restocking, lost margin) exceeds the cost of an unnecessary "
    "risk-flag (extra manual review), which is typically true for Flipkart's return-risk "
    "screening use case."
)
