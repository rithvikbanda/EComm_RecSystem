import pandas as pd
import numpy as np
import os

def load_and_clean_data(file_path):
    """
    Load and clean the retail data.
    - Robust CSV decoding (utf-8, then latin1)
    - Drop null CustomerID and cast to int
    - Strip strings but keep true NaN (avoid turning NaN into the string "nan")
    - Remove cancellations (InvoiceNo starting with 'C' or Quantity <= 0)
    - Add TotalSpend
    """
    # Robust CSV load
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin1")

    print(f"Original data shape: {df.shape}")

    # Clean
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Keep true NaNs for Description/Country
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["Description"] = df["Description"].astype("string").str.strip()
    df["Country"] = df["Country"].astype("string").str.strip()

    # Remove cancellations
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[df["Quantity"] > 0]

    # Spend
    df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]

    print(f"After cleaning: {df.shape}")
    return df

def _majority_value(s: pd.Series):
    """
    Return the statistical mode; if none, fall back to first non-null,
    else NaN.
    """
    m = s.mode(dropna=True)
    if len(m):
        return m.iloc[0]
    s_nonnull = s.dropna()
    return s_nonnull.iloc[0] if len(s_nonnull) else np.nan

def aggregate_customer_data(df: pd.DataFrame):
    """
    Aggregate purchases per customer (vectorized, fast).
    - products: sorted unique product descriptions
    - total_spent: sum of TotalSpend
    - purchase_frequency: number of unique invoices (transactions)
    - avg_spend_per_product: total_spent / unique_products_count
    - country: majority country
    """
    g = df.groupby("CustomerID", as_index=True)

    products = g["Description"].agg(lambda s: sorted(set(s.dropna())))
    total_spent = g["TotalSpend"].sum().round(2)
    purchase_frequency = g["InvoiceNo"].nunique()
    unique_products_count = g["Description"].nunique(dropna=True)
    country = g["Country"].apply(_majority_value)

    avg_spend_per_product = (
        total_spent / unique_products_count.replace(0, np.nan)
    ).fillna(0).round(2)

    profiles_df = pd.DataFrame({
        "customer_id": products.index.astype(int),
        "products": products.values,
        "country": country.values,
        "total_spent": total_spent.values,
        "purchase_frequency": purchase_frequency.values,
        "avg_spend_per_product": avg_spend_per_product.values,
        "unique_products_count": unique_products_count.values
    }).reset_index(drop=True)

    return profiles_df.to_dict(orient="records")

def main():
    file_path = "data.csv"  # raw file
    cleaned_data = load_and_clean_data(file_path)

    customer_profiles = aggregate_customer_data(cleaned_data)

    # Save cleaned data for embedding step
    output_path = "cleaned_data.csv"
    cleaned_data.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned data to {output_path}")

    print(f"Created {len(customer_profiles)} customer profiles")
    if customer_profiles:
        print("\nSample customer profile:")
        print(customer_profiles[0])

    return customer_profiles, cleaned_data

if __name__ == "__main__":
    profiles, data = main()
