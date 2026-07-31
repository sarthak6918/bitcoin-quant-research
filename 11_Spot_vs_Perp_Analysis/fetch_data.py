"""
Fetch BTC/USDT spot and BTC/USDT perpetual futures OHLCV from Binance's
public REST APIs (no key needed), 2019-01-01 -> present, daily and hourly.

Spot:    https://api.binance.com/api/v3/klines
Futures: https://fapi.binance.com/fapi/v1/klines  (USDT-M perpetual, BTCUSDT)
Perp futures on Binance launched 2019-09-08, so daily data before that does
not exist for the perp and is simply absent (not zero-filled).
"""
import time
import requests
import pandas as pd

SPOT_URL = "https://api.binance.com/api/v3/klines"
FUT_URL = "https://fapi.binance.com/fapi/v1/klines"
START = "2019-01-01"
END = "2026-07-29"

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def fetch(url, symbol, interval, start, end):
    # Binance spot caps klines at 1000/call; USDT-M futures allows 1500.
    limit = 1500 if "fapi" in url else 1000
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    rows = []
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cur,
                 "endTime": end_ms, "limit": limit}
        for attempt in range(6):
            try:
                r = requests.get(url, params=params, timeout=20)
                r.raise_for_status()
                batch = r.json()
                break
            except requests.exceptions.RequestException as e:
                wait = 2 * (attempt + 1)
                print(f"  retry {attempt+1}/6 after {e.__class__.__name__}, sleeping {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError("failed after 6 retries")
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        if last_open <= cur:
            break
        cur = last_open + 1
        if len(batch) < limit:
            break
        time.sleep(0.3)
    df = pd.DataFrame(rows, columns=COLS)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    import os
    # spot hourly already exists, verified, in 02_Master_Hourly_OHLCV -- reuse
    # it instead of re-fetching identical public data.
    reuse_path = "../02_Master_Hourly_OHLCV_With_Regime/btc_1h_ohlcv_2017_2026_with_regime.csv"
    if os.path.exists(reuse_path):
        spot_h = pd.read_csv(reuse_path, parse_dates=["timestamp"])
        spot_h = spot_h[spot_h["timestamp"] >= START][["timestamp", "open", "high", "low", "close", "volume"]]
        spot_h.to_csv("spot_hourly.csv", index=False)
        print(f"reused spot_hourly.csv from 02_Master_Hourly_OHLCV: {len(spot_h)} rows, "
              f"{spot_h['timestamp'].min()} -> {spot_h['timestamp'].max()}")
        jobs = [("spot", SPOT_URL, "BTCUSDT", [("1d", "daily")])]
    else:
        jobs = [("spot", SPOT_URL, "BTCUSDT", [("1d", "daily"), ("1h", "hourly")]),
               ("perp", FUT_URL, "BTCUSDT", [("1d", "daily"), ("1h", "hourly")])]

    for label, url, symbol, intervals in jobs:
        for interval, tag in intervals:
            print(f"fetching {label} {tag} ...")
            df = fetch(url, symbol, interval, START, END)
            print(f"  {label} {tag}: {len(df)} rows, {df['timestamp'].min()} -> {df['timestamp'].max()}")
            df.to_csv(f"{label}_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
