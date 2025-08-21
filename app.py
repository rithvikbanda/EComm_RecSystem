#!/usr/bin/env python3
# streamlit run app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Try to import your upgraded rag_search; if not found, provide light fallbacks.
try:
    import rag_search as rs  # uses your upgraded script if in PYTHONPATH / same dir
except Exception as e:
    rs = None

    # ---- Light inline fallbacks (subset of rag_search functionality) ----
    import faiss
    from sentence_transformers import SentenceTransformer

    def _l2_normalize(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + eps
        return mat / norms

    class _Catalog:
        def __init__(
            self,
            products_meta: pd.DataFrame,
            product_emb: np.ndarray,
            customers_meta: Optional[pd.DataFrame],
            customer_emb: Optional[np.ndarray],
            cleaned: Optional[pd.DataFrame],
        ):
            self.products_meta = products_meta
            self.product_emb = _l2_normalize(product_emb.astype("float32"))
            self.customers_meta = customers_meta
            self.customer_emb = _l2_normalize(customer_emb.astype("float32")) if customer_emb is not None else None
            self.cleaned = cleaned
            self.index = faiss.IndexFlatIP(self.product_emb.shape[1])
            self.index.add(self.product_emb)

    @st.cache_data(show_spinner=False)
    def _encode_texts(model_name: str, device: str, texts: Tuple[str, ...]) -> np.ndarray:
        model = SentenceTransformer(model_name, device=None if device == "cpu" else device)
        arr = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, device=device, show_progress_bar=False)
        return arr.astype("float32")

    def _search_text(cat: _Catalog, model: str, device: str, query: str, k: int) -> pd.DataFrame:
        q = _encode_texts(model, device, (query,))
        D, I = cat.index.search(q, k)
        rows = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            row = cat.products_meta.iloc[int(idx)].to_dict()
            row["score"] = float(score)
            rows.append(row)
        return pd.DataFrame(rows)

    def _similar_products(cat: _Catalog, stock_code: str, k: int) -> pd.DataFrame:
        pm = cat.products_meta.copy()
        pm["_idx"] = np.arange(len(pm))
        sub = pm[pm["StockCode"].astype(str) == str(stock_code)]
        if sub.empty:
            return pd.DataFrame()
        idx = int(sub["_idx"].iloc[0])
        q = cat.product_emb[idx][None, :]
        D, I = cat.index.search(q.astype("float32"), k + 1)
        rows = []
        for score, j in zip(D[0], I[0]):
            if j == idx:
                continue
            row = cat.products_meta.iloc[int(j)].to_dict()
            row["score"] = float(score)
            rows.append(row)
            if len(rows) >= k:
                break
        return pd.DataFrame(rows)

    def _recs_for_customer(cat: _Catalog, customer_id: int, k: int, filter_purchased: bool, diversify: bool, mmr_lambda: float) -> pd.DataFrame:
        if cat.customers_meta is None or cat.customer_emb is None:
            return pd.DataFrame()
        cm = cat.customers_meta.copy()
        cm["_idx"] = np.arange(len(cm))
        row = cm[cm["CustomerID"].astype(int) == int(customer_id)]
        if row.empty:
            return pd.DataFrame()
        cidx = int(row["_idx"].iloc[0])
        q = cat.customer_emb[cidx]

        mask = np.ones(len(cat.products_meta), dtype=bool)
        if filter_purchased and cat.cleaned is not None and "CustomerID" in cat.cleaned and "StockCode" in cat.cleaned:
            purchased = set(cat.cleaned.loc[cat.cleaned["CustomerID"].astype("Int64") == int(customer_id), "StockCode"].astype(str).tolist())
            if purchased:
                mask = ~cat.products_meta["StockCode"].astype(str).isin(purchased).values

        filtered_emb = cat.product_emb[mask]
        if filtered_emb.shape[0] == 0:
            return pd.DataFrame()

        index = faiss.IndexFlatIP(filtered_emb.shape[1])
        index.add(filtered_emb.astype("float32"))
        D, I = index.search(q[None, :].astype("float32"), k * 5)

        idx_map = np.where(mask)[0]
        cand_idx = idx_map[I[0]]
        cand_meta = cat.products_meta.iloc[cand_idx].copy()
        cand_meta["score"] = D[0]

        if not diversify:
            return cand_meta.nlargest(k, "score").reset_index(drop=True)

        # Simple MMR
        def mmr(qv: np.ndarray, X: np.ndarray, lam: float, top_k: int):
            if X.shape[0] == 0:
                return []
            qv = qv.reshape(1, -1)
            sim_to_q = (X @ qv.T).ravel()
            sims = X @ X.T
            selected, candidates = [], list(range(X.shape[0]))
            while candidates and len(selected) < top_k:
                if not selected:
                    best = int(np.argmax(sim_to_q[candidates]))
                    selected.append(candidates.pop(best))
                    continue
                max_to_sel = np.max(sims[candidates][:, selected], axis=1)
                scores = lam * sim_to_q[candidates] - (1 - lam) * max_to_sel
                best = int(np.argmax(scores))
                selected.append(candidates.pop(best))
            return selected

        cap = min(200, filtered_emb.shape[0])
        sel = mmr(q, filtered_emb[I[0][:cap]], lam=mmr_lambda, top_k=k)
        selected_idx = I[0][:cap][sel]
        out_idx = idx_map[selected_idx]
        out = cat.products_meta.iloc[out_idx].copy().reset_index(drop=True)
        out["score"] = (filtered_emb[selected_idx] @ q).astype("float32")
        return out

# ---------------- UI ----------------
st.set_page_config(page_title="E‑commerce Recommender Demo", layout="wide")

st.title("🛒 E‑commerce Recommender Demo")
st.caption("Semantic product search • Similar items • Customer recommendations")

with st.sidebar:
    st.header("Artifacts & Settings")
    products_meta_path = st.text_input("Products meta CSV", "artifacts/products_meta.csv")
    product_emb_path = st.text_input("Product embeddings NPY", "artifacts/product_embeddings.npy")
    customers_meta_path = st.text_input("Customers meta CSV (optional)", "artifacts/customers_meta.csv")
    customer_emb_path = st.text_input("Customer embeddings NPY (optional)", "artifacts/customer_embeddings.npy")
    cleaned_path = st.text_input("Cleaned data CSV (optional)", "data/cleaned_data.csv")

    model_name = st.text_input("Text model (for search-text)", "sentence-transformers/all-MiniLM-L6-v2")
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    top_k = st.number_input("Top K", min_value=1, max_value=100, value=10, step=1)

    load_btn = st.button("Load Catalog", type="primary")

@st.cache_resource(show_spinner=True)
def load_catalog_cached(pmeta: str, pemb: str, cmeta: str, cemb: str, cleaned: str):
    pm = pd.read_csv(pmeta)
    pe = np.load(pemb)
    cm = pd.read_csv(cmeta) if cmeta and Path(cmeta).exists() else None
    ce = np.load(cemb) if cemb and Path(cemb).exists() else None
    cd = pd.read_csv(cleaned) if cleaned and Path(cleaned).exists() else None
    if rs is not None:
        # Use your full-featured loader
        return rs.load_catalog(Path(pmeta), Path(pemb), Path(cmeta) if cm is not None else None, Path(cemb) if ce is not None else None, Path(cleaned) if cd is not None else None)
    else:
        return _Catalog(pm, pe, cm, ce, cd)

if load_btn:
    try:
        cat = load_catalog_cached(products_meta_path, product_emb_path, customers_meta_path, customer_emb_path, cleaned_path)
        st.success("Catalog loaded.")
        st.session_state["cat_loaded"] = True
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        st.session_state["cat_loaded"] = False

if st.session_state.get("cat_loaded"):
    cat = load_catalog_cached(products_meta_path, product_emb_path, customers_meta_path, customer_emb_path, cleaned_path)

    tab1, tab2, tab3 = st.tabs(["🔎 Text Search", "🔁 Similar Product", "🎯 Customer Recs"])

    with tab1:
        q = st.text_input("Search query", "Red Umbrella")
        if st.button("Search"):
            try:
                if rs is not None:
                    df = rs.search_by_text(cat, q, model_name=model_name, top_k=top_k, device=device)
                else:
                    df = _search_text(cat, model=model_name, device=device, query=q, k=top_k)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Search failed: {e}")

    with tab2:
        pm = cat.products_meta
        # Build a small selector; if large, let user type
        stock_codes = pm["StockCode"].astype(str).tolist()
        default_sc = stock_codes[0] if stock_codes else ""
        sc = st.text_input("StockCode", default_sc)
        if st.button("Find similar"):
            try:
                if rs is not None:
                    df = rs.similar_products(cat, sc, top_k=top_k)
                else:
                    df = _similar_products(cat, sc, k=top_k)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Similar search failed: {e}")

    with tab3:
        cust_id = st.number_input("CustomerID", min_value=0, value=17850, step=1)
        filter_purchased = st.checkbox("Filter purchased items", value=True)
        diversify = st.checkbox("Diversify results (MMR)", value=True)
        mmr_lambda = st.slider("MMR λ (relevance vs diversity)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        if st.button("Recommend"):
            try:
                if rs is not None:
                    df = rs.recommendations_for_customer(
                        cat,
                        customer_id=int(cust_id),
                        top_k=int(top_k),
                        filter_purchased=bool(filter_purchased),
                        diversify=bool(diversify),
                        mmr_lambda=float(mmr_lambda),
                    )
                else:
                    df = _recs_for_customer(cat, int(cust_id), int(top_k), bool(filter_purchased), bool(diversify), float(mmr_lambda))
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Recommendation failed: {e}")
else:
    st.info("Use the sidebar to load your artifacts, then try the tabs above.")
