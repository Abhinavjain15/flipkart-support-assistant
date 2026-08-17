from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from preprocessing import load_splits, build_preprocessor

X_train, X_test, y_train, y_test = load_splits()

rf_pipe = Pipeline(
    [
        ("preprocess", build_preprocessor()),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42)),
    ]
)

param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [6, 10, None],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(rf_pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
test_proba = best_rf.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_proba)

print("Best params:", grid.best_params_)
print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
print(f"Test ROC-AUC:    {test_auc:.4f}")
print(f"Gap (CV - test): {abs(grid.best_score_ - test_auc):.4f}")
