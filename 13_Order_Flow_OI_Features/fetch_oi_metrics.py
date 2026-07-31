"""Full-history open-interest / long-short-ratio metrics for BTCUSDT perp.

Source: data.binance.vision daily 'metrics' dumps (5-minute granularity,
available from ~2021-01 onward — earlier days 404 and are skipped).
Each day's zip is ~10KB; full history is small enough to keep raw days too,
but we resample to hourly immediately and only persist that.

Resumable: already-processed days are tracked in metrics_checkpoint.txt,
so re-running after an interruption just continues.

Output: oi_metrics_hourly.csv
"""
import sys

import pandas as pd

from common import append_checkpoint, daterange, fetch_daily_zip_csv, load_checkpoint

SYMBOL = "BTCUSDT"
START = "2021-01-01"
CHECKPOINT = "metrics_checkpoint.txt"
OUT = "oi_metrics_hourly.csv"


def hourly_agg(df: pd.DataFrame) -> pd.DataFrame:
    df["create_time"] = pd.to_datetime(df["create_time"])
    df = df.set_index("create_time")
    hourly = df.resample("1h").agg({
        "sum_open_interest": "last",
        "sum_open_interest_value": "last",
        "count_toptrader_long_short_ratio": "mean",
        "sum_toptrader_long_short_ratio": "mean",
        "count_long_short_ratio": "mean",
        "sum_taker_long_short_vol_ratio": "mean",
    })
    return hourly.reset_index().rename(columns={"create_time": "timestamp"})


def main(end_date: str):
    done = load_checkpoint(CHECKPOINT)
    wrote_header = False
    try:
        with open(OUT) as f:
            wrote_header = bool(f.readline())
    except FileNotFoundError:
        pass

    for day in daterange(START, end_date):
        if day in done:
            continue
        df = fetch_daily_zip_csv("metrics", SYMBOL, day)
        if df is None:
            print(f"{day}: not available, skipping")
            append_checkpoint(CHECKPOINT, day)
            continue
        hourly = hourly_agg(df)
        hourly.to_csv(OUT, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        append_checkpoint(CHECKPOINT, day)
        print(f"{day}: {len(hourly)} hourly rows appended")


if __name__ == "__main__":
    end = sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    main(end)
