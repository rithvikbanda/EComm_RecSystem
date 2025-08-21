from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os

# === Config ===
DATA_DIR = "data"
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
PRODUCTS_EMB_PATH = os.path.join(DATA_DIR, "product_embeddings.npy")
PRODUCTS_META_PATH = os.path.join(DATA_DIR, "products_meta.csv")
MODEL_NAME = "all-MiniLM-L6-v2"

# === Load cleaned data ===
if not os.path.exists(CLEANED_DATA_PATH):
    raise FileNotFoundError(f"{CLEANED_DATA_PATH} not found. Run cleandata.py first.")

cleaned_data = pd.read_csv(CLEANED_DATA_PATH)

# Create unique product table
products_df = (
    cleaned_data.groupby("StockCode")
    .agg({"Description": "first"})
    .reset_index()
)

# === Model ===
print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# Combine code + description
product_texts = (products_df["StockCode"] + " " + products_df["Description"]).tolist()

# === Embed products ===
print(f"Embedding {len(product_texts)} products...")
product_embeddings = model.encode(
    product_texts, 
    convert_to_numpy=True, 
    normalize_embeddings=True
)

# Save outputs
np.save(PRODUCTS_EMB_PATH, product_embeddings)
products_df.to_csv(PRODUCTS_META_PATH, index=False)

print(f" Saved embeddings to {PRODUCTS_EMB_PATH}")
print(f" Saved product metadata to {PRODUCTS_META_PATH}")
print("Product embeddings shape:", product_embeddings.shape)
