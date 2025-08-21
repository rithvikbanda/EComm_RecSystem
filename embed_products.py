#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Iterable, Set

import numpy as np
import pandas as pd

# You need: pip install sentence-transformers
from sentence_transformers import SentenceTransformer


REQUIRED_COLS: Set[str] = {"StockCode", "Description"}


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


def validate_schema(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLS) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def encode_texts(model: SentenceTransformer, texts, batch_size: int = 512, device: str = "cpu") -> np.ndarray:
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        arr = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
            show_progress_bar=False,
        )
        embs.append(arr.astype("float32"))
    return np.vstack(embs) if embs else np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")


def build_products(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only necessary columns; drop NA, dedupe, stable order
    out = (
        df.loc[:, ["StockCode", "Description"]]
        .dropna(subset=["StockCode"])
        .copy()
    )
    out["StockCode"] = out["StockCode"].astype(str).str.strip()
    out["Description"] = out["Description"].astype("string").str.strip()
    out = out.dropna(subset=["Description"])
    out = out.drop_duplicates(subset=["StockCode"]).sort_values("StockCode").reset_index(drop=True)
    out["product_text"] = (out["StockCode"].astype(str) + " " + out["Description"].fillna("")).str.strip()
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed unique products from cleaned transactions.")
    p.add_argument("--input", type=Path, default=Path("data/cleaned_data.csv"), help="Cleaned CSV path")
    p.add_argument("--meta-out", type=Path, default=Path("artifacts/products_meta.csv"), help="Where to write product metadata CSV")
    p.add_argument("--emb-out", type=Path, default=Path("artifacts/product_embeddings.npy"), help="Where to write product embeddings (.npy)")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name")
    p.add_argument("--batch-size", type=int, default=512, help="Encoding batch size")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    df = pd.read_csv(args.input)
    validate_schema(df)

    products_df = build_products(df)
    logging.info("Unique products: %d", len(products_df))

    model = SentenceTransformer(args.model, device=args.device if args.device != "cpu" else None)
    dim = model.get_sentence_embedding_dimension()
    logging.info("Loaded model %s (dim=%d)", args.model, dim)

    embeddings = encode_texts(model, products_df["product_text"].tolist(), batch_size=args.batch_size, device=args.device)
    assert embeddings.shape[0] == len(products_df) and embeddings.shape[1] == dim

    # Ensure output dirs
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.emb_out.parent.mkdir(parents=True, exist_ok=True)

    products_df.to_csv(args.meta_out, index=False)
    np.save(args.emb_out, embeddings)

    manifest = {
        "model_name": args.model,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meta_path": str(args.meta_out),
        "emb_path": str(args.emb_out),
        "meta_sha256": hash_file(args.meta_out),
        "emb_sha256": hash_file(args.emb_out),
    }
    man_path = args.meta_out.with_suffix(".manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logging.info("Wrote products_meta -> %s", args.meta_out)
    logging.info("Wrote product_embeddings -> %s", args.emb_out)
    logging.info("Wrote manifest -> %s", man_path)


if __name__ == "__main__":
    main()
