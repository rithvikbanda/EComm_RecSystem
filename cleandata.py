#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Set

import numpy as np
import pandas as pd

# ================= Config =================
DEFAULT_DATA_DIR = Path("data")
DEFAULT_INPUT = DEFAULT_DATA_DIR / "data.csv"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "cleaned_data.csv"

REQUIRED_COLS: Set[str] = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}

# ============== Logging ==============
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


# ============== I/O Helpers ==============
def read_csv_robust(path: Path) -> pd.DataFrame:
    """Try UTF-8 then latin1 to avoid hard failures on mixed encodings."""
    last_err = None
    for enc in ("utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err  # type: ignore[misc]


def validate_schema(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLS) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


# ============== Cleaning Steps ==============
def clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply production-leaning cleaning rules compatible with downstream steps."""
    validate_schema(raw)

    df = raw.copy()

    # Basic hygiene
    before = len(df)
    df = df.drop_duplicates()
    logging.info("Dropped %d duplicate rows", before - len(df))

    # Drop rows missing key IDs
    df = df.dropna(subset=["CustomerID", "StockCode", "InvoiceNo"])

    # Normalize types
    df["CustomerID"] = df["CustomerID"].astype("int64", errors="ignore")
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype("string").str.strip()
    df["Country"] = df["Country"].astype("string").str.strip()

    # Parse dates; drop rows with unparseable dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", utc=False)
    bad_dates = df["InvoiceDate"].isna().sum()
    if bad_dates:
        logging.warning("Dropping %d rows with bad InvoiceDate", bad_dates)
        df = df.dropna(subset=["InvoiceDate"])

    # Remove cancellations (InvoiceNo starting with 'C')
    before = len(df)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    logging.info("Removed %d cancellation rows", before - len(df))

    # Coerce numeric and filter invalids
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    bad_numeric = df["Quantity"].isna().sum() + df["UnitPrice"].isna().sum()
    if bad_numeric:
        logging.warning("Found %d rows with non-numeric Quantity/UnitPrice -> dropping", bad_numeric)
    df = df.dropna(subset=["Quantity", "UnitPrice"])

    # Positive quantities & prices only
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    logging.info("Filtered %d rows with non-positive qty/price", before - len(df))

    # Light winsorization to cap extreme outliers on UnitPrice
    # (keeps demo stable while avoiding distorting downstream embeddings)
    cap = df["UnitPrice"].quantile(0.999)
    if pd.notna(cap) and cap > 0:
        df.loc[df["UnitPrice"] > cap, "UnitPrice"] = cap

    # Compute spend
    df["TotalSpend"] = (df["Quantity"] * df["UnitPrice"]).round(2)

    # Sort for determinism (useful for tests and reproducibility)
    df = df.sort_values(["CustomerID", "InvoiceDate", "InvoiceNo", "StockCode"]).reset_index(drop=True)

    return df


# ============== CLI ==============
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean retail transactions into a stable CSV for embeddings/recs.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to raw CSV (default: data/data.csv)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write cleaned CSV (default: data/cleaned_data.csv)")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found.")

    logging.info("Loading raw CSV: %s", args.input)
    raw = read_csv_robust(args.input)
    logging.info("Raw shape: %s", raw.shape)

    cleaned = clean_dataframe(raw)
    logging.info("Cleaned shape: %s", cleaned.shape)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    logging.info("Wrote cleaned CSV -> %s", args.output)

    # Useful preview
    try:
        summary = cleaned.agg({"Quantity": ["sum", "mean"], "UnitPrice": ["mean"], "TotalSpend": ["sum", "mean"]})
        print("\n=== Data Summary ===")
        print(summary.to_string())
    except Exception:
        pass


if __name__ == "__main__":
    main()
