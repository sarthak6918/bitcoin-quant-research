"""Full-history BTCUSDT perp funding rate — free, no key, ~2900 rows total.

Output: funding_rate_full_history.csv (fundingTime, fundingRate, markPrice)
"""
import time

import pandas as pd
import requests

OUT = "funding_rate_full_history.csv"
URL = "https://fapi.binance.com/fapi/v1/fundingRate"
START = pd.Timestamp("2019-09-01")


def fetch_all():
    rows = []
    start_ms = int(START.timestamp() * 1000)
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": "BTCUSDT", "startTime": cur, "limit": 1000}
        for attempt in range(5):
            try:
                r = requests.get(URL, params=params, timeout=20)
                r.raise_for_status()
                batch = r.json()
                break
            except requests.exceptions.RequestException:
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"failed at cur={cur}")
        if not batch:
            break
        rows.extend(batch)
        last_time = batch[-1]["fundingTime"]
        if last_time <= cur:
            break
        cur = last_time + 1
        print(f"  ...{pd.to_datetime(last_time, unit='ms')}  ({len(rows)} rows so far)")
        time.sleep(0.2)
    return rows


if __name__ == "__main__":
    print("Fetching full BTCUSDT funding rate history...")
    rows = fetch_all()
    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["markPrice"] = pd.to_numeric(df["markPrice"], errors="coerce")
    df = df.drop_duplicates("fundingTime").sort_values("fundingTime")
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows -> {OUT}  ({df.fundingTime.min()} -> {df.fundingTime.max()})")
