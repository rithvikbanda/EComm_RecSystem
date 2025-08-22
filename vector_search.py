#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import faiss                      # pip install faiss-cpu
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer  # for text queries


# =========================
# Utilities
# =========================
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def l2_normalize(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + eps
    return mat / norms


def cosine_ip_search(index: faiss.Index, q: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """FAISS IndexFlatIP: expects normalized vectors to approximate cosine."""
    if q.ndim == 1:
        q = q[None, :]
    D, I = index.search(q.astype(np.float32), top_k)
    return D, I


def load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# =========================
# Data holders
# =========================
@dataclass
class Catalog:
    products_meta: pd.DataFrame
    product_emb: np.ndarray
    customers_meta: Optional[pd.DataFrame]
    customer_emb: Optional[np.ndarray]
    cleaned: Optional[pd.DataFrame]
    product_index: faiss.Index


# =========================
# Loading
# =========================
def load_catalog(
    products_meta_path: Path,
    product_emb_path: Path,
    customers_meta_path: Optional[Path] = None,
    customer_emb_path: Optional[Path] = None,
    cleaned_path: Optional[Path] = None,
) -> Catalog:
    # Products
    products_meta = pd.read_csv(products_meta_path)
    product_emb = np.load(product_emb_path)
    if product_emb.ndim != 2:
        raise ValueError("Product embeddings must be 2D")
    d = product_emb.shape[1]

    # Validate product shapes
    if len(products_meta) != product_emb.shape[0]:
        raise ValueError("products_meta rows != product_embeddings rows")

    # Normalize product vectors for IP
    product_emb = l2_normalize(product_emb.astype("float32"))

    # Build FAISS IP index
    index = faiss.IndexFlatIP(d)
    index.add(product_emb)

    # Customers (optional)
    customers_meta = None
    customer_emb = None
    if customers_meta_path and customer_emb_path:
        customers_meta = pd.read_csv(customers_meta_path)
        customer_emb = np.load(customer_emb_path).astype("float32")
        if customer_emb.ndim != 2 or customer_emb.shape[1] != d:
            raise ValueError("Customer embeddings must be 2D and have same dim as product embeddings")
        customer_emb = l2_normalize(customer_emb)

    # Cleaned (optional) for purchase filtering
    cleaned = None
    if cleaned_path and cleaned_path.exists():
        cleaned = pd.read_csv(cleaned_path)
        # Light normalize types; tolerate missing columns
        for col in ("CustomerID", "StockCode"):
            if col in cleaned.columns:
                if col == "CustomerID":
                    cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").astype("Int64")
                else:
                    cleaned[col] = cleaned[col].astype(str)

    return Catalog(
        products_meta=products_meta,
        product_emb=product_emb,
        customers_meta=customers_meta,
        customer_emb=customer_emb,
        cleaned=cleaned,
        product_index=index,
    )


# =========================
# Search primitives
# =========================
def encode_text_query(texts: List[str], model_name: str, device: str = "cpu") -> np.ndarray:
    model = SentenceTransformer(model_name, device=None if device == "cpu" else device)
    q = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device,
        show_progress_bar=False,
    ).astype("float32")
    return q


def search_by_text(cat: Catalog, query: str, model_name: str, top_k: int, device: str = "cpu") -> pd.DataFrame:
    q = encode_text_query([query], model_name=model_name, device=device)
    D, I = cosine_ip_search(cat.product_index, q, top_k)
    rows = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        row = cat.products_meta.iloc[int(idx)].to_dict()
        row["score"] = float(score)
        rows.append(row)
    return pd.DataFrame(rows)


def similar_products(cat: Catalog, stock_code: str, top_k: int) -> pd.DataFrame:
    # Find product index
    pm = cat.products_meta
    pm["_idx"] = np.arange(len(pm))
    sub = pm[pm["StockCode"].astype(str) == str(stock_code)]
    if sub.empty:
        raise ValueError(f"StockCode {stock_code} not found in products_meta")
    idx = int(sub["_idx"].iloc[0])
    q = cat.product_emb[idx]
    D, I = cosine_ip_search(cat.product_index, q, top_k + 1)  # +1 to include itself
    rows = []
    for score, j in zip(D[0], I[0]):
        if j == idx:
            continue
        row = cat.products_meta.iloc[int(j)].to_dict()
        row["score"] = float(score)
        rows.append(row)
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)


def mmr(query_vec: np.ndarray, cand_vecs: np.ndarray, lambda_: float = 0.5, top_k: int = 10) -> List[int]:
    """
    Maximal Marginal Relevance for diversification.
    Returns indices into cand_vecs of selected items.
    """
    if cand_vecs.shape[0] == 0:
        return []
    selected = []
    candidates = list(range(cand_vecs.shape[0]))
    # Precompute sims
    q = query_vec.reshape(1, -1).astype("float32")
    sim_to_q = (cand_vecs @ q.T).ravel()
    sims = cand_vecs @ cand_vecs.T  # cosine since normalized
    while candidates and len(selected) < top_k:
        if not selected:
            best = int(np.argmax(sim_to_q[candidates]))
            selected.append(candidates.pop(best))
            continue
        # compute max sim to already selected
        max_to_sel = np.max(sims[candidates][:, selected], axis=1)
        # mmr score
        scores = lambda_ * sim_to_q[candidates] - (1 - lambda_) * max_to_sel
        best = int(np.argmax(scores))
        selected.append(candidates.pop(best))
    return selected


def recommendations_for_customer(
    cat: Catalog,
    customer_id: int,
    top_k: int = 10,
    filter_purchased: bool = True,
    diversify: bool = True,
    mmr_lambda: float = 0.5,
) -> pd.DataFrame:
    if cat.customers_meta is None or cat.customer_emb is None:
        raise ValueError("Customer embeddings/meta not loaded. Provide --customers-meta/--customer-emb.")

    # Find customer row
    cm = cat.customers_meta
    cm["_idx"] = np.arange(len(cm))
    row = cm[cm["CustomerID"].astype(int) == int(customer_id)]
    if row.empty:
        raise ValueError(f"CustomerID {customer_id} not found in customers_meta")
    cidx = int(row["_idx"].iloc[0])
    q = cat.customer_emb[cidx]

    # Filter candidates if requested (exclude purchased products)
    mask = np.ones(len(cat.products_meta), dtype=bool)
    if filter_purchased and cat.cleaned is not None:
        purchased = set(
            cat.cleaned.loc[cat.cleaned["CustomerID"] == customer_id, "StockCode"].astype(str).tolist()
        )
        if purchased:
            mask = ~cat.products_meta["StockCode"].astype(str).isin(purchased).values

    # Run search over filtered set
    filtered_emb = cat.product_emb[mask]
    if filtered_emb.shape[0] == 0:
        return pd.DataFrame(columns=list(cat.products_meta.columns) + ["score"])

    # Simple top-k search first
    index = faiss.IndexFlatIP(filtered_emb.shape[1])
    index.add(filtered_emb.astype("float32"))
    D, I = cosine_ip_search(index, q, top_k * 5)  # overfetch; we'll MMR or slice later

    # Build candidate frame
    idx_map = np.where(mask)[0]
    cand_idx = idx_map[I[0]]
    scores = D[0]
    cand_meta = cat.products_meta.iloc[cand_idx].copy()
    cand_meta["score"] = scores

    if diversify:
        # apply MMR to top ~min(200, len(candidates))
        cap = min(200, len(cand_meta))
        sel = mmr(q, filtered_emb[I[0][:cap]], lambda_=mmr_lambda, top_k=top_k)
        selected_idx = I[0][:cap][sel]
        out_idx = idx_map[selected_idx]
        out = cat.products_meta.iloc[out_idx].copy().reset_index(drop=True)
        out["score"] = (filtered_emb[selected_idx] @ q).astype("float32")
        return out
    else:
        return cand_meta.nlargest(top_k, "score").reset_index(drop=True)


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG-style semantic search & recommendations over product/customer embeddings.")
    p.add_argument("--products-meta", type=Path, default=Path("artifacts/products_meta.csv"))
    p.add_argument("--product-emb", type=Path, default=Path("artifacts/product_embeddings.npy"))
    p.add_argument("--customers-meta", type=Path, default=Path("artifacts/customers_meta.csv"))
    p.add_argument("--customer-emb", type=Path, default=Path("artifacts/customer_embeddings.npy"))
    p.add_argument("--cleaned", type=Path, default=Path("data/cleaned_data.csv"))
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Model for text query encoding")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--log-level", default="INFO")

    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("search-text", help="Semantic search over products by text query")
    s1.add_argument("--query", required=True)

    s2 = sub.add_parser("similar-product", help="Find similar products to a stock code")
    s2.add_argument("--stock-code", required=True)

    s3 = sub.add_parser("recs-for-customer", help="Recommend products for a customer")
    s3.add_argument("--customer-id", type=int, required=True)
    s3.add_argument("--no-filter-purchased", action="store_true")
    s3.add_argument("--no-diversify", action="store_true")
    s3.add_argument("--mmr-lambda", type=float, default=0.5)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    # Load catalog & indices
    cat = load_catalog(
        products_meta_path=args.products_meta,
        product_emb_path=args.product_emb,
        customers_meta_path=args.customers_meta if args.customers_meta and args.customers_meta.exists() else None,
        customer_emb_path=args.customer_emb if args.customer_emb and args.customer_emb.exists() else None,
        cleaned_path=args.cleaned if args.cleaned and args.cleaned.exists() else None,
    )

    if args.cmd == "search-text":
        df = search_by_text(cat, args.query, model_name=args.model, top_k=args.top_k, device=args.device)
        print(df.head(args.top_k).to_string(index=False))

    elif args.cmd == "similar-product":
        df = similar_products(cat, args.stock_code, top_k=args.top_k)
        print(df.to_string(index=False))

    elif args.cmd == "recs-for-customer":
        df = recommendations_for_customer(
            cat,
            customer_id=args.customer_id,
            top_k=args.top_k,
            filter_purchased=not args.no_filter_purchased,
            diversify=not args.no_diversify,
            mmr_lambda=args.mmr_lambda,
        )
        print(df.to_string(index=False))

    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
