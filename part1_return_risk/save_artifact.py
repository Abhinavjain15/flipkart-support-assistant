import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, recall_score, precision_score
from preprocessing import load_splits, build_preprocessor

X_train, X_test, y_train, y_test = load_splits()

# Final chosen model: tuned RF from Task 6 (max_depth=6, n_estimators=100)
final_pipe = Pipeline(
    [
        ("preprocess", build_preprocessor()),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=100, max_depth=6, class_weight="balanced", random_state=42
            ),
        ),
    ]
)
final_pipe.fit(X_train, y_train)

# Re-run Task 5's threshold-sweep procedure, but on THIS model's own predict_proba
rf_proba = final_pipe.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.10, 0.901, 0.02)

best_f1, t_star_rf, best_recall, best_precision = -1, None, None, None
for t in thresholds:
    preds_t = (rf_proba >= t).astype(int)
    f1_t = f1_score(y_test, preds_t, pos_label=1, zero_division=0)
    if f1_t > best_f1:
        best_f1 = f1_t
        t_star_rf = round(t, 2)
        best_recall = recall_score(y_test, preds_t, pos_label=1, zero_division=0)
        best_precision = precision_score(y_test, preds_t, pos_label=1, zero_division=0)

print(f"t*_rf (F1-maximizing threshold on RF's own predict_proba): {t_star_rf:.2f}")
print(f"  F1={best_f1:.4f}  Recall={best_recall:.4f}  Precision={best_precision:.4f}")

os.makedirs("../models", exist_ok=True)
joblib.dump(final_pipe, "../models/return_risk_model.pkl")
print("\nSaved: models/return_risk_model.pkl")

# Sanity check: reload and confirm predict_proba matches
reloaded = joblib.load("../models/return_risk_model.pkl")
reloaded_proba = reloaded.predict_proba(X_test)[:, 1]
assert np.allclose(rf_proba, reloaded_proba), "Reloaded model output mismatch!"
print("Reload check passed: predict_proba matches original fitted pipeline.")

# Persist t*_rf for Part 3's tool to consume directly (avoids recomputation drift)
with open("../models/t_star_rf.txt", "w") as f:
    f.write(str(t_star_rf))
print(f"Saved: models/t_star_rf.txt -> {t_star_rf}")
