#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import numpy as np
import pandas as pd

REQUIRED_CLEANED: Set[str] = {
    "InvoiceNo",
    "StockCode",
    "Quantity",
    "InvoiceDate",
    "CustomerID",
}

REQUIRED_PRODUCTS: Set[str] = {"StockCode", "Description", "product_text"}


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_schema(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build customer embeddings from product embeddings with advanced weighting.")
    p.add_argument("--cleaned", type=Path, default=Path("data/cleaned_data.csv"))
    p.add_argument("--products-meta", type=Path, default=Path("artifacts/products_meta.csv"))
    p.add_argument("--product-emb", type=Path, default=Path("artifacts/product_embeddings.npy"))
    p.add_argument("--customers-meta-out", type=Path, default=Path("artifacts/customers_meta.csv"))
    p.add_argument("--customer-emb-out", type=Path, default=Path("artifacts/customer_embeddings.npy"))
    p.add_argument("--tau-days", type=float, default=180.0, help="Recency decay time constant in days")
    p.add_argument("--min-products", type=int, default=1, help="Minimum unique products required to emit a customer vector")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    # Load inputs
    if not args.cleaned.exists():
        raise FileNotFoundError(args.cleaned)
    if not args.products_meta.exists():
        raise FileNotFoundError(args.products_meta)
    if not args.product_emb.exists():
        raise FileNotFoundError(args.product_emb)

    tx = pd.read_csv(args.cleaned)
    validate_schema(tx, REQUIRED_CLEANED)
    products = pd.read_csv(args.products_meta)
    validate_schema(products, REQUIRED_PRODUCTS)
    prod_emb = np.load(args.product_emb)
    dim = prod_emb.shape[1]
    logging.info("Loaded %d product vectors (dim=%d)", prod_emb.shape[0], dim)

    # Align product order
    products = products.drop_duplicates(subset=["StockCode"]).sort_values("StockCode").reset_index(drop=True)
    if prod_emb.shape[0] != len(products):
        raise ValueError("product_embeddings rows != products_meta rows (after dedupe/sort).")

    # Build mapping StockCode -> row index in embeddings
    stock_to_idx: Dict[str, int] = {sc: i for i, sc in enumerate(products["StockCode"].astype(str))}

    # Prepare transactions
    tx = tx.dropna(subset=["CustomerID", "StockCode", "Quantity", "InvoiceDate"]).copy()
    tx["CustomerID"] = tx["CustomerID"].astype("int64")
    tx["StockCode"] = tx["StockCode"].astype(str)
    tx["Quantity"] = pd.to_numeric(tx["Quantity"], errors="coerce").astype("float32")
    tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"], errors="coerce")

    tx = tx.dropna(subset=["InvoiceDate"])
    tx = tx[tx["Quantity"] > 0]

    # Map to product indices; drop unknown stock codes
    tx["p_idx"] = tx["StockCode"].map(stock_to_idx)
    tx = tx.dropna(subset=["p_idx"]).copy()
    tx["p_idx"] = tx["p_idx"].astype("int32")

    # Popularity by product (unique customers)
    pop = (
        tx.groupby("StockCode")["CustomerID"].nunique().astype("float32")
        .reindex(products["StockCode"]).fillna(1.0).values
    )

    # Recency decay
    max_ts = tx["InvoiceDate"].max()
    days_ago = (max_ts - tx["InvoiceDate"]).dt.days.clip(lower=0).astype("int32")
    tau = float(args.tau_days)
    tx["time_decay"] = np.exp(-days_ago / tau).astype("float32")

    # Weighting: log(1+qty) * 1/log(2+popularity) * exp(-days/tau)
    qty = tx["Quantity"].values.astype("float32")
    pidx = tx["p_idx"].values
    gpop = pop[pidx]
    decay = tx["time_decay"].values
    weights = np.log1p(qty) * (1.0 / np.log(2.0 + gpop)) * decay
    txw = pd.DataFrame({"CustomerID": tx["CustomerID"].values, "p_idx": pidx, "w": weights})

    # Collapse duplicates (customer, product)
    txw = txw.groupby(["CustomerID", "p_idx"], as_index=False)["w"].sum()

    # Filter customers with too few unique products
    cust_counts = txw.groupby("CustomerID")["p_idx"].nunique()
    keep_customers = cust_counts[cust_counts >= int(args.min_products)].index
    txw = txw[txw["CustomerID"].isin(keep_customers)]

    # Materialize customer vectors
    cust_ids = txw["CustomerID"].unique()
    cust_ids.sort()
    cust_vecs = np.zeros((len(cust_ids), dim), dtype="float32")
    for i, cid in enumerate(cust_ids):
        sub = txw[txw["CustomerID"] == cid]
        v = prod_emb[sub["p_idx"].values]
        w = sub["w"].values.astype("float32")
        s = w.sum()
        if s <= 0:
            continue
        w = w / s
        cust_vecs[i] = (v * w[:, None]).sum(axis=0)
        # Normalize
        n = np.linalg.norm(cust_vecs[i]) + 1e-8
        cust_vecs[i] /= n

    # Build customers meta
    meta = (
        tx.groupby("CustomerID")
        .agg(
            n_txns=("InvoiceNo", "nunique"),
            n_unique_products=("StockCode", "nunique"),
            total_qty=("Quantity", "sum"),
            last_purchase_ts=("InvoiceDate", "max"),
        )
        .reset_index()
    )
    meta = meta[meta["CustomerID"].isin(cust_ids)]
    meta = meta.sort_values("CustomerID").reset_index(drop=True)

    # Ensure output dirs
    args.customers_meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.customer_emb_out.parent.mkdir(parents=True, exist_ok=True)

    # Save
    np.save(args.customer_emb_out, cust_vecs)
    meta.to_csv(args.customers_meta_out, index=False)

    # Manifest
    products_manifest_path = args.products_meta.with_suffix(".manifest.json")
    products_manifest_sha = hash_file(products_manifest_path) if products_manifest_path.exists() else None

    manifest = {
        "dim": int(cust_vecs.shape[1]),
        "count": int(cust_vecs.shape[0]),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "customer_emb_path": str(args.customer_emb_out),
        "customers_meta_path": str(args.customers_meta_out),
        "customer_emb_sha256": hash_file(args.customer_emb_out),
        "customers_meta_sha256": hash_file(args.customers_meta_out),
        "source_products_manifest_sha256": products_manifest_sha,
        "tau_days": tau,
        "min_products": int(args.min_products),
        "weighting": "log1p(qty) * 1/log(2+popularity) * exp(-days/tau)",
    }
    man_path = args.customers_meta_out.with_suffix(".manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logging.info("Wrote customers_meta -> %s", args.customers_meta_out)
    logging.info("Wrote customer_embeddings -> %s", args.customer_emb_out)
    logging.info("Wrote manifest -> %s", man_path)
    logging.info("Total customers embedded: %d", len(cust_ids))


if __name__ == "__main__":
    main()
