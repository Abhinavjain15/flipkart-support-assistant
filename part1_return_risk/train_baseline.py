from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from preprocessing import load_splits, build_preprocessor

X_train, X_test, y_train, y_test = load_splits()

baseline = Pipeline(
    [
        ("preprocess", build_preprocessor()),
        ("clf", DummyClassifier(strategy="most_frequent", random_state=42)),
    ]
)
baseline.fit(X_train, y_train)  # preprocessor fit on train only, not test
preds = baseline.predict(X_test)

acc = accuracy_score(y_test, preds)
f1_pos = f1_score(y_test, preds, pos_label=1)

print(f"Baseline accuracy: {acc:.4f}")
print(f"Baseline F1 (returned=1): {f1_pos:.4f}")
print(
    "\nWhy this is misleading: the dataset's ~77% majority-class share means always "
    "predicting 'not returned' yields high accuracy while recall for the class that "
    "actually matters (returned=1) is exactly 0 -- the model never identifies a single "
    "return. This is the 'high accuracy, zero recall' trap: accuracy is not aligned with "
    "the real business problem (catching returns), and without comparing to this baseline "
    "a mediocre model could look deceptively good on accuracy alone."
)
