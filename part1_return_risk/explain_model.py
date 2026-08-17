import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from preprocessing import load_splits, build_preprocessor

X_train, X_test, y_train, y_test = load_splits()

# Best params from train_rf.py GridSearchCV (deterministic given fixed seeds)
preprocessor = build_preprocessor()
rf_pipe = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=100, max_depth=6, class_weight="balanced", random_state=42
            ),
        ),
    ]
)
rf_pipe.fit(X_train, y_train)

fitted_preprocessor = rf_pipe.named_steps["preprocess"]
rf_clf = rf_pipe.named_steps["clf"]
feature_names = [
    name.split("__", 1)[1] for name in fitted_preprocessor.get_feature_names_out()
]

# --- Impurity-based importance ---
impurity_scores = rf_clf.feature_importances_
impurity_ranked = sorted(zip(feature_names, impurity_scores), key=lambda x: -x[1])
top5_impurity = impurity_ranked[:5]

print("=== Top 5 features (impurity-based .feature_importances_) ===")
for name, score in top5_impurity:
    print(f"{name:30s} {score:.4f}")

# --- Permutation importance (on transformed test features, RF scored on roc_auc) ---
X_test_transformed = fitted_preprocessor.transform(X_test)
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

perm_result = permutation_importance(
    rf_clf,
    X_test_transformed,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)
perm_scores = dict(zip(feature_names, perm_result.importances_mean))

print("\n=== Same top-5 features under permutation_importance (ROC-AUC drop) ===")
comparison = []
for name, impurity_score in top5_impurity:
    perm_score = perm_scores[name]
    comparison.append((name, impurity_score, perm_score))
    print(f"{name:30s} impurity={impurity_score:.4f}  permutation={perm_score:.4f}")

# Identify which top-5 features lose most importance under permutation
# (rank drop = normalized impurity rank position vs normalized permutation rank position)
perm_ranked_all = sorted(perm_scores.items(), key=lambda x: -x[1])
perm_rank_of = {name: i for i, (name, _) in enumerate(perm_ranked_all)}
biggest_drop = max(comparison, key=lambda row: perm_rank_of[row[0]])

print(f"\nFeature losing most importance under permutation: {biggest_drop[0]}")
print(
    f"  (impurity rank #{[n for n,_ in top5_impurity].index(biggest_drop[0])+1}, "
    f"permutation rank #{perm_rank_of[biggest_drop[0]]+1} of {len(feature_names)})"
)
print(
    "Impurity-based importance can overrate a noisy continuous feature because RF split-selection "
    "favors high-cardinality continuous columns purely due to having more possible split points, "
    "regardless of whether those splits generalize to unseen data."
)

# --- Interpretation of top-5 (dynamic lookup by feature substring) ---
INTERPRETATIONS = {
    "num_previous_returns": "a customer's historical return count is a direct behavioral signal of return propensity.",
    "payment_method_COD": "cash-on-delivery removes upfront payment commitment, correlating with lower purchase intent and higher return/refusal rates.",
    "price_inr": "higher-priced items carry more scrutiny and buyer's remorse risk, raising return likelihood.",
    "discount_pct": "heavily discounted items are often impulse buys, which are more prone to being returned.",
    "customer_tenure_days": "longer-tenured customers have more established purchase habits and historically lower return rates.",
    "product_category_Apparel": "apparel has inherent fit/size uncertainty driving returns.",
    "product_category_Footwear": "footwear has inherent fit/size uncertainty driving returns.",
    "delivery_days": "longer delivery windows can increase order cancellation/return likelihood as urgency fades.",
    "num_previous_orders": "order history size reflects customer engagement level and predictability of behavior.",
}

print("\n=== Interpretation of top-5 features ===")
for name, _ in top5_impurity:
    explanation = next(
        (v for k, v in INTERPRETATIONS.items() if k in name),
        "plausibly correlates with return behavior via the data-generating process's dependency on this variable.",
    )
    print(f"- {name}: {explanation}")

git_names_top5 = [n for n, _ in top5_impurity]
print(f"\nTop-5 names for acceptance-check reference: {git_names_top5}")
