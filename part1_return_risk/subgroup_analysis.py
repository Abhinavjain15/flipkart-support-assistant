import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score
from preprocessing import load_splits, build_preprocessor

X_train, X_test, y_train, y_test = load_splits()

rf_pipe = Pipeline(
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
rf_pipe.fit(X_train, y_train)
test_preds = rf_pipe.predict(X_test)

results = X_test.copy()
results["y_true"] = y_test.values
results["y_pred"] = test_preds


def subgroup_metrics(df, group_col):
    rows = []
    for group_val, sub in df.groupby(group_col):
        r = recall_score(sub["y_true"], sub["y_pred"], pos_label=1, zero_division=0)
        p = precision_score(sub["y_true"], sub["y_pred"], pos_label=1, zero_division=0)
        rows.append(
            {
                group_col: group_val,
                "n": len(sub),
                "recall": round(r, 4),
                "precision": round(p, 4),
            }
        )
    return pd.DataFrame(rows)


overall_recall = recall_score(results["y_true"], results["y_pred"], pos_label=1)
overall_precision = precision_score(results["y_true"], results["y_pred"], pos_label=1)
print(f"Overall test recall={overall_recall:.4f}  precision={overall_precision:.4f}\n")

cat_table = subgroup_metrics(results, "product_category")
print("=== By product_category ===")
print(cat_table.to_string(index=False))

pay_table = subgroup_metrics(results, "payment_method")
print("\n=== By payment_method ===")
print(pay_table.to_string(index=False))

print(
    "\nWeak subgroup: identify the row(s) above with recall meaningfully below "
    f"{overall_recall:.4f} (overall). "
    "Proposed fix: train a category-specific decision threshold for that subgroup "
    "(sweep threshold within just that category's held-out predictions rather than "
    "using one global threshold) so its recall/precision matches the weakest-category "
    "target instead of being diluted by the global cut point -- this is more targeted "
    "than 'collect more data' since it fixes a calibration mismatch, not a data volume gap."
)
