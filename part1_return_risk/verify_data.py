import pandas as pd

df = pd.read_csv("orders_dataset.csv")

print("Rows:", df.shape[0], "| Columns:", df.shape[1])
print("Overall return rate:", round(df["returned"].mean(), 4))

missing_pct = df["rating_given"].isna().mean() * 100
print(f"rating_given missing: {missing_pct:.2f}%")

print("\nReturn rate by product_category:")
print(df.groupby("product_category")["returned"].mean().round(4))

print("\nReturn rate by payment_method:")
print(df.groupby("payment_method")["returned"].mean().round(4))

# MAR evidence: missingness rate conditional on payment_method
cod_missing = df.loc[df["payment_method"] == "COD", "rating_given"].isna().mean()
non_cod_missing = df.loc[df["payment_method"] != "COD", "rating_given"].isna().mean()
gap_pp = (cod_missing - non_cod_missing) * 100

print(f"\nCOD missing rate: {cod_missing*100:.2f}%")
print(f"Non-COD missing rate: {non_cod_missing*100:.2f}%")
print(f"Gap: {gap_pp:.2f} percentage points")
print(
    "\nClassification: MAR (missing-at-random, conditional on observed payment_method).\n"
    f"Evidence: COD orders are missing rating_given at {cod_missing*100:.2f}% vs "
    f"{non_cod_missing*100:.2f}% for non-COD, a {gap_pp:.2f}pp gap explained by the "
    "generator's missing_mask depending on payment_method. Not MCAR (the gap shows a real "
    "dependency on an observed column). Not MNAR (missingness does not depend on the "
    "unobserved rating_given value itself, only on payment_method)."
)
