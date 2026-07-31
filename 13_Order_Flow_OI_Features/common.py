"""Shared helpers for the free-tier microstructure ETL (Binance Vision + REST).

Design constraint: this machine has ~8.7GB free disk. Every fetcher here
streams a day's zip into memory, aggregates it down to a handful of hourly
rows, and discards the raw bytes immediately -- nothing raw ever touches
disk. Only the small aggregated CSVs are persisted.
"""
import io
import time
import zipfile

import pandas as pd
import requests

VISION_BASE = "https://data.binance.vision/data/futures/um/daily"
UA = {"User-Agent": "Mozilla/5.0 (research-etl)"}


def daterange(start, end):
    d = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while d <= end:
        yield d.strftime("%Y-%m-%d")
        d += pd.Timedelta(days=1)


def fetch_daily_zip_csv(kind: str, symbol: str, day: str, retries: int = 4) -> pd.DataFrame | None:
    """Download one day's Binance Vision zip fully into memory, return the
    single CSV inside as a DataFrame. Returns None on 404 (day not published
    yet / not available). Never writes to disk."""
    url = f"{VISION_BASE}/{kind}/{symbol}/{symbol}-{kind}-{day}.zip"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as f:
                    return pd.read_csv(f)
        except requests.exceptions.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


def load_checkpoint(path) -> set:
    try:
        with open(path) as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()


def append_checkpoint(path, day: str):
    with open(path, "a") as f:
        f.write(day + "\n")
