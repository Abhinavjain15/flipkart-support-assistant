# Return-Risk Tool Call

**User:** Check the return risk for this order.
Order: {"price_inr": 1800, "discount_pct": 45.0, "customer_tenure_days": 90, "num_previous_orders": 2, "num_previous_returns": 1, "delivery_distance_km": 320.0, "delivery_days": 6, "is_weekend_order": 1, "rating_given": null, "product_category": "Apparel", "payment_method": "COD"}

**Agent:**
```json
{
  "answer": "This order has an estimated return probability of 60.41%, classified as Medium risk (anchored to t*_rf=0.46).",
  "source": "return_risk_tool",
  "confidence": 0.6041
}
```

