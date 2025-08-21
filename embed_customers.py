import pandas as pd
import numpy as np
import os

# === Config ===
DATA_DIR = "data"
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
PRODUCTS_META_PATH = os.path.join(DATA_DIR, "products_meta.csv")
PRODUCTS_EMB_PATH = os.path.join(DATA_DIR, "product_embeddings.npy")
CUSTOMERS_META_PATH = os.path.join(DATA_DIR, "customers_meta.csv")
CUSTOMERS_EMB_PATH = os.path.join(DATA_DIR, "customer_embeddings.npy")

# === Checks ===
if not os.path.exists(CLEANED_DATA_PATH):
    raise FileNotFoundError(f"{CLEANED_DATA_PATH} not found. Run cleandata.py first.")
if not os.path.exists(PRODUCTS_META_PATH) or not os.path.exists(PRODUCTS_EMB_PATH):
    raise FileNotFoundError("Run embed_products.py first to generate product embeddings.")

# === Load data ===
cleaned_data = pd.read_csv(CLEANED_DATA_PATH)
products_df = pd.read_csv(PRODUCTS_META_PATH)
product_embeddings = np.load(PRODUCTS_EMB_PATH)

# Map StockCode → product index
stockcode_to_idx = {code: i for i, code in enumerate(products_df["StockCode"])}

# Build customer embeddings
customer_vectors = []
customer_ids = []

for cust_id, group in cleaned_data.groupby("CustomerID"):
    vectors = []
    weights = []
    for _, row in group.iterrows():
        sc = row["StockCode"]
        if sc in stockcode_to_idx:
            idx = stockcode_to_idx[sc]
            vectors.append(product_embeddings[idx])
            weights.append(row["Quantity"])
    if vectors:
        vectors = np.array(vectors)
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()  # normalize weights
        customer_vector = np.average(vectors, axis=0, weights=weights)
        customer_vectors.append(customer_vector)
        customer_ids.append(cust_id)

customer_embeddings = np.vstack(customer_vectors)

# Save to disk
np.save(CUSTOMERS_EMB_PATH, customer_embeddings)
pd.DataFrame({"CustomerID": customer_ids}).to_csv(CUSTOMERS_META_PATH, index=False)

print(f"✅ Saved {len(customer_ids)} customer embeddings to {CUSTOMERS_EMB_PATH}")
print(f"✅ Saved customer metadata to {CUSTOMERS_META_PATH}")
print("Customer embeddings shape:", customer_embeddings.shape)
