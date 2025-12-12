from __future__ import annotations
import os, yaml, datetime as dt, pytz
import pandas as pd

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_tickers(path: str) -> list[str]:
    """Read tickers from file, filtering comments and deduping."""
    seen = set()
    tickers = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ticker = ln.strip().upper()
            # Skip empty lines, comments, and triple-quote markers
            if not ticker or ticker.startswith("#") or ticker.startswith("'''") or ticker.startswith('"""'):
                continue
            # Dedupe while preserving order
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    return tickers

def new_york_now() -> dt.datetime:
    return dt.datetime.now(tz=pytz.timezone("US/Eastern"))

def round2(x) -> float:
    try:
        return float(f"{float(x):.2f}")
    except Exception:
        return 0.0
