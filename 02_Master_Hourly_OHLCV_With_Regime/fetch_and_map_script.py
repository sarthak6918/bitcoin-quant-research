"""
fetch_1h_and_map_regime.py — Pull continuous 1H BTC/USDT OHLCV and map the
already-fitted daily HMM regime onto it.
=============================================================================
Does NOT refit the HMM. Uses the locked K=5 model's decoded daily output
(btc_regime_features.csv, produced by btc_regime_hmm.py) and Andrew's
existing map_regime_to_1h() to forward-fill the daily regime onto fresh
continuous hourly bars, with the same one-day lag already built into that
function (a day's regime is only "available" once that daily bar closes —
no lookahead).

Two steps:
  1. Fetch continuous BTC/USDT 1H OHLCV from Binance, paginated from
     --start_date to now (same public REST endpoint used for daily bars,
     just a different interval / larger row count — ~78k rows for the
     full 2017-2026 history vs ~3.2k daily rows).
  2. Merge-asof the daily regime columns (filtered_state,
     filtered_state_label, filtered_prob_state0..K-1) onto every 1H bar,
     using each bar's own trailing daily state.

Usage:
    python fetch_1h_and_map_regime.py \
        --daily_features_csv btc_regime_output_adx_k5/btc_regime_features.csv \
        --start_date 2017-08-17 \
        --output_dir btc_regime_output_adx_k5
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import btc_regime_hmm as regime_lib  # reuse map_regime_to_1h — no refit here

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL_1H = "1h"
KLINES_LIMIT = 1000


def fetch_1h_ohlcv(start_date: str, cache_path: Path, symbol: str = SYMBOL) -> pd.DataFrame:
    """
    Fetch continuous 1H BTC/USDT OHLCV from Binance, paginated from
    start_date to now. Mirrors fetch_daily_ohlcv's pagination logic exactly,
    just with a 1-hour bar interval, a much larger row count, and a
    'timestamp' column instead of 'date' (map_regime_to_1h auto-detects
    'timestamp', 'Date/Time', 'date', or 'datetime').
    """
    if cache_path.exists():
        print(f"  ✓ Using cached 1H OHLCV: {cache_path}")
        return pd.read_csv(cache_path, parse_dates=["timestamp"])

    start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    now_ts = int(pd.Timestamp.utcnow().timestamp() * 1000)
    interval_ms = 60 * 60 * 1000  # 1 hour

    all_rows = []
    cursor = start_ts
    print(f"  Fetching {symbol} {INTERVAL_1H} klines from {start_date} ... "
          f"(this is a much bigger pull than daily — expect ~{(now_ts-start_ts)//interval_ms:,} bars)")
    while cursor < now_ts:
        params = {
            "symbol": symbol,
            "interval": INTERVAL_1H,
            "startTime": cursor,
            "limit": KLINES_LIMIT,
        }
        for attempt in range(5):
            try:
                r = requests.get(BINANCE_KLINES_URL, params=params, timeout=20)
                if r.status_code == 200:
                    batch = r.json()
                    break
                else:
                    print(f"    ✗ HTTP {r.status_code}: {r.text[:150]} (retry {attempt+1}/5)")
                    time.sleep(2)
            except Exception as e:
                print(f"    ✗ Request error: {e} (retry {attempt+1}/5)")
                time.sleep(2)
        else:
            raise RuntimeError("Failed to fetch 1H klines after 5 retries")

        if not batch:
            break

        all_rows.extend(batch)
        last_open_time = batch[-1][0]
        cursor = last_open_time + interval_ms
        if len(all_rows) % 20000 < KLINES_LIMIT:
            print(f"    ... {len(all_rows):,} bars so far, last = "
                  f"{pd.to_datetime(last_open_time, unit='ms')}")
        if len(batch) < KLINES_LIMIT:
            break
        time.sleep(0.15)  # be polite to the API — this is ~10x more calls than the daily fetch

    if not all_rows:
        raise RuntimeError("No 1H data returned from Binance — check network/start_date")

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]] \
        .drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    df.to_csv(cache_path, index=False)
    print(f"  ✓ Saved 1H OHLCV → {cache_path}  ({len(df):,} bars, "
          f"{df['timestamp'].min()} → {df['timestamp'].max()})")
    return df


def main():
    ap = argparse.ArgumentParser(description="Fetch continuous 1H BTC OHLCV and map the locked daily regime onto it")
    ap.add_argument("--daily_features_csv", required=True,
                     help="Path to btc_regime_features.csv from the locked HMM run "
                          "(must contain date, filtered_state, filtered_state_label, "
                          "filtered_prob_state* columns)")
    ap.add_argument("--start_date", default="2017-08-17",
                     help="Should match (or predate) the daily model's start_date")
    ap.add_argument("--output_dir", default="./btc_regime_output")
    ap.add_argument("--symbol", default=SYMBOL)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FETCH 1H OHLCV + MAP LOCKED DAILY REGIME")
    print("=" * 70)

    print("\n[1/3] Loading locked daily regime decode ...")
    daily_regime_df = pd.read_csv(args.daily_features_csv, parse_dates=["date"])
    required_cols = {"date", "filtered_state", "filtered_state_label"}
    missing = required_cols - set(daily_regime_df.columns)
    if missing:
        raise ValueError(f"{args.daily_features_csv} is missing columns: {missing}. "
                          f"Is this the btc_regime_features.csv from the locked K=5 run?")
    k_found = sum(1 for c in daily_regime_df.columns if c.startswith("filtered_prob_state"))
    print(f"  ✓ Loaded {len(daily_regime_df)} daily rows, "
          f"{daily_regime_df['date'].min().date()} → {daily_regime_df['date'].max().date()}, "
          f"K={k_found}")

    print("\n[2/3] Fetching continuous 1H OHLCV ...")
    fetch_1h_ohlcv(args.start_date, out_dir / "btc_1h_ohlcv.csv", symbol=args.symbol)

    print("\n[3/3] Mapping daily regime onto 1H bars (1-day lag, no lookahead) ...")
    merged = regime_lib.map_regime_to_1h(
        daily_regime_df,
        str(out_dir / "btc_1h_ohlcv.csv"),
        out_dir / "btc_1h_with_regime.csv",
    )

    n_unlabeled = merged["filtered_state"].isna().sum()
    print(f"\n  Rows without a regime label yet (feature warmup / pre-history): {n_unlabeled:,} "
          f"of {len(merged):,} ({n_unlabeled/len(merged)*100:.2f}%)")

    print("\n" + "=" * 70)
    print("DONE.")
    print(f"1H OHLCV with regime → {out_dir / 'btc_1h_with_regime.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
