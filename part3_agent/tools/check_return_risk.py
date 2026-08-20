"""
Tool: check_return_risk
Loads Part 1's tuned Random Forest pipeline and returns a return-probability
score plus a risk bucket anchored to t*_rf (the F1-maximizing threshold
computed on THIS model's own predict_proba, not a fixed 0.3/0.6 split).
"""

import joblib
import pandas as pd

MODEL_PATH = "../../models/return_risk_model.pkl"
T_STAR_PATH = "../../models/t_star_rf.txt"

_model = None
_t_star_rf = None

EXPECTED_FEATURES = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
    "product_category",
    "payment_method",
]


def _load_artifacts():
    global _model, _t_star_rf
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _t_star_rf is None:
        with open(T_STAR_PATH) as f:
            _t_star_rf = float(f.read().strip())
    return _model, _t_star_rf


def check_return_risk(order_features: dict) -> dict:
    model, t_star_rf = _load_artifacts()

    missing = [f for f in EXPECTED_FEATURES if f not in order_features]
    if missing:
        raise ValueError(f"Missing required order features: {missing}")

    X = pd.DataFrame([order_features])[EXPECTED_FEATURES]
    proba = float(
        model.predict_proba(X)[0, 1]
    )  # real call to Part 1's saved RF, not a stand-in

    # Bucket cut points anchored to t*_rf (Part 1 Task 9), not fixed values.
    low_cut = t_star_rf
    high_cut = t_star_rf + 0.15
    if proba < low_cut:
        bucket = "Low"
    elif proba >= high_cut:
        bucket = "High"
    else:
        bucket = "Medium"

    return {
        "return_probability": round(proba, 4),
        "risk_bucket": bucket,
        "t_star_rf": t_star_rf,
        "cut_points": {"low_cut": round(low_cut, 2), "high_cut": round(high_cut, 2)},
    }


if __name__ == "__main__":
    sample_order = {
        "price_inr": 1800,
        "discount_pct": 45.0,
        "customer_tenure_days": 90,
        "num_previous_orders": 2,
        "num_previous_returns": 1,
        "delivery_distance_km": 320.0,
        "delivery_days": 6,
        "is_weekend_order": 1,
        "rating_given": None,
        "product_category": "Apparel",
        "payment_method": "COD",
    }
    result = check_return_risk(sample_order)
    print("Sample order risk assessment:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print(
        f"\nCut points: Low if prob < {result['cut_points']['low_cut']}, "
        f"High if prob >= {result['cut_points']['high_cut']}, else Medium. "
        f"Anchored to t*_rf={result['t_star_rf']} from Part 1 Task 9."
    )
