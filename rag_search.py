#!/usr/bin/env python3
import os
import json
import re
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ===== OpenAI (LLM intent parser) =====
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ================== Config ==================
DATA_DIR = "data"
PRODUCTS_META_PATH = os.path.join(DATA_DIR, "products_meta.csv")
PRODUCTS_EMB_PATH = os.path.join(DATA_DIR, "product_embeddings.npy")
CUSTOMERS_META_PATH = os.path.join(DATA_DIR, "customers_meta.csv")
CUSTOMERS_EMB_PATH = os.path.join(DATA_DIR, "customer_embeddings.npy")
CLEANED_DATA_PATH  = os.path.join(DATA_DIR, "cleaned_data.csv")

# Use the SAME model you used to build product/customer embeddings
MODEL_NAME = "all-MiniLM-L6-v2"

# LLM settings (used only for intent parsing)
LLM_MODEL = "gpt-4o-mini"
# If you hardcoded your key, you can put it here; otherwise set the env var.
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None

SYSTEM_PROMPT = """You are a query parser for a product recommendation CLI.
Return ONLY a compact JSON object with keys:
- intent: one of ["customer_recs","text_search"]
- customer_id: integer or null
- extra_text: string (may be empty)
Rules:
- If the text includes a 4-7 digit number that looks like a customer id, prefer intent="customer_recs".
- If there is no customer id, use intent="text_search".
No explanations, only valid JSON.
"""

# ================== Load data ==================
for p in [PRODUCTS_META_PATH, PRODUCTS_EMB_PATH, CUSTOMERS_META_PATH, CUSTOMERS_EMB_PATH, CLEANED_DATA_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file: {p}. Run clean_data.py, embed_products.py, embed_customers.py first.")

products_df         = pd.read_csv(PRODUCTS_META_PATH)
product_embeddings  = np.load(PRODUCTS_EMB_PATH).astype("float32")
customers_df        = pd.read_csv(CUSTOMERS_META_PATH)
customer_embeddings = np.load(CUSTOMERS_EMB_PATH).astype("float32")
cleaned_data        = pd.read_csv(CLEANED_DATA_PATH)

# ================== FAISS (products) ==================
d = product_embeddings.shape[1]          # 384 for MiniLM
index = faiss.IndexFlatIP(d)             # inner product == cosine if normalized
index.add(product_embeddings)

# ================== Query encoder (MiniLM) ==================
print(f"Loading embedding model for queries: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

def embed_query(text: str) -> np.ndarray:
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype("float32")  # (1, d)

# ================== Retrieval ==================
def search_products_by_text(query: str, top_k: int = 5):
    q_emb = embed_query(query)
    D, I = index.search(q_emb, top_k)
    results = products_df.iloc[I[0]].copy()
    return results, D[0]

def recommend_for_customer(customer_id: int, top_k: int = 5, filter_purchased: bool = True):
    if customer_id not in customers_df["CustomerID"].values:
        return None, None
    idx = customers_df.index[customers_df["CustomerID"] == customer_id][0]
    q_vec = customer_embeddings[idx:idx+1]  # already MiniLM space
    D, I = index.search(q_vec, top_k + 20)  # over-fetch for filtering
    recs = products_df.iloc[I[0]].copy()

    if filter_purchased:
        bought = set(cleaned_data.loc[cleaned_data["CustomerID"] == customer_id, "StockCode"])
        recs = recs[~recs["StockCode"].isin(bought)]

    return recs.head(top_k), None

# ================== LLM intent parsing ==================
def parse_query_llm(user_text: str) -> dict:
    """LLM parse with regex fallback."""
    if client is None:
        return _regex_fallback(user_text)
    try:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ]
        resp = client.chat.completions.create(model=LLM_MODEL, messages=msgs, temperature=0)
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)

        intent = data.get("intent")
        if intent not in {"customer_recs", "text_search"}:
            raise ValueError("bad intent")

        cid = data.get("customer_id")
        if cid is not None:
            try:
                data["customer_id"] = int(cid)
            except Exception:
                data["customer_id"] = None

        if "extra_text" not in data:
            data["extra_text"] = ""
        return data
    except Exception:
        return _regex_fallback(user_text)

def _regex_fallback(user_text: str) -> dict:
    """If LLM is unavailable or fails, extract a 4–7 digit ID if present, else text search."""
    m = re.search(r"\b(\d{4,7})\b", user_text)
    if m:
        cid = int(m.group(1))
        extra = re.sub(r"\b(\d{4,7})\b", "", user_text).strip()
        return {"intent": "customer_recs", "customer_id": cid, "extra_text": extra}
    return {"intent": "text_search", "customer_id": None, "extra_text": user_text}

# ================== CLI ==================
if __name__ == "__main__":
    print(" Type anything:")
    print("   - Plain English:  product for holding clothes")
    print("   - Customer recs:  recommend for customer 12303")
    print("   Type 'exit' to quit.\n")

    while True:
        user_input = input("Query: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        parsed = parse_query_llm(user_input)
        intent = parsed.get("intent")
        cid    = parsed.get("customer_id")
        extra  = (parsed.get("extra_text") or "").strip()

        if intent == "customer_recs" and cid is not None:
            results, _ = recommend_for_customer(cid, top_k=5, filter_purchased=True)
            if results is None or results.empty:
                print(f"❌ No customer with ID {cid} found or no unseen products.")
            else:
                print(f"\nTop recommendations for customer {cid}:")
                for _, r in results.iterrows():
                    print(f"  {r['StockCode']} - {r['Description']}")
        else:
            query_text = extra if extra else user_input
            results, scores = search_products_by_text(query_text, top_k=5)
            print(f"\nTop products for: “{query_text}”")
            for (i, r), s in zip(results.iterrows(), scores):
                print(f"  {r['StockCode']} - {r['Description']}  (score: {s:.3f})")
