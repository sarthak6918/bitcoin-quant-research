"""Order-flow imbalance features from BTCUSDT perp aggTrades, full free tier.

Source: data.binance.vision daily 'aggTrades' dumps (every executed trade,
side-tagged via is_buyer_maker). Available from ~2020-01 onward.

Disk-safety: each day's zip (~10-20MB) is downloaded straight into memory,
parsed, aggregated to ~24 hourly feature rows, and the ~1-1.5M raw trade
rows are discarded before moving to the next day. Nothing raw is ever
written to disk. This machine has ~8.7GB free -- do not change this to
"download everything then process."

Resumable: processed days are tracked in orderflow_checkpoint.txt. Re-running
with the same --start/--end continues where it left off.

Convention (Binance aggTrades): is_buyer_maker == True means the BUYER was
the resting/maker order, i.e. the trade was SELLER-initiated (aggressive
sell). is_buyer_maker == False means the trade was BUYER-initiated
(aggressive buy). "Taker" flow below always refers to the aggressor side.

Usage:
    python build_orderflow_features.py --start 2025-07-01 --end 2026-07-30
    (defaults to the trailing 12 months if no args given, per the
    recommendation to prove signal on a modest window before paying for
    full history / order-book depth)

Output: orderflow_hourly_features.csv
"""
import argparse

import numpy as np
import pandas as pd

from common import append_checkpoint, daterange, fetch_daily_zip_csv, load_checkpoint

SYMBOL = "BTCUSDT"
CHECKPOINT = "orderflow_checkpoint.txt"
OUT = "orderflow_hourly_features.csv"
LARGE_TRADE_BTC = 1.0  # qty threshold for "large" (whale-sized) taker trades


def hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    df["transact_time"] = pd.to_datetime(df["transact_time"], unit="ms")
    df["hour"] = df["transact_time"].dt.floor("h")

    is_sell_taker = df["is_buyer_maker"]          # aggressive sell
    is_buy_taker = ~df["is_buyer_maker"]           # aggressive buy

    df["buy_qty"] = np.where(is_buy_taker, df["quantity"], 0.0)
    df["sell_qty"] = np.where(is_sell_taker, df["quantity"], 0.0)
    df["buy_notional"] = df["buy_qty"] * df["price"]
    df["sell_notional"] = df["sell_qty"] * df["price"]
    df["is_large"] = df["quantity"] >= LARGE_TRADE_BTC
    df["large_buy_qty"] = np.where(is_buy_taker & df["is_large"], df["quantity"], 0.0)
    df["large_sell_qty"] = np.where(is_sell_taker & df["is_large"], df["quantity"], 0.0)

    g = df.groupby("hour")
    out = pd.DataFrame({
        "taker_buy_vol": g["buy_qty"].sum(),
        "taker_sell_vol": g["sell_qty"].sum(),
        "taker_buy_notional": g["buy_notional"].sum(),
        "taker_sell_notional": g["sell_notional"].sum(),
        "trade_count": g.size(),
        "buy_trade_count": g["buy_qty"].apply(lambda s: (s > 0).sum()),
        "large_buy_vol": g["large_buy_qty"].sum(),
        "large_sell_vol": g["large_sell_qty"].sum(),
        "vwap": g.apply(lambda d: (d["price"] * d["quantity"]).sum() / d["quantity"].sum()),
        "price_close": g["price"].last(),
    })
    out["net_taker_flow"] = out["taker_buy_vol"] - out["taker_sell_vol"]
    total_vol = out["taker_buy_vol"] + out["taker_sell_vol"]
    out["ofi"] = out["net_taker_flow"] / total_vol.replace(0, np.nan)
    out["taker_buy_ratio"] = out["taker_buy_vol"] / total_vol.replace(0, np.nan)
    out["avg_trade_size"] = total_vol / out["trade_count"]
    out["large_trade_net_flow"] = out["large_buy_vol"] - out["large_sell_vol"]
    out["sell_trade_count"] = out["trade_count"] - out["buy_trade_count"]
    out["trade_count_imbalance"] = (
        (out["buy_trade_count"] - out["sell_trade_count"]) / out["trade_count"]
    )
    return out.reset_index().rename(columns={"hour": "timestamp"})


def main(start: str, end: str):
    done = load_checkpoint(CHECKPOINT)
    try:
        with open(OUT) as f:
            wrote_header = bool(f.readline())
    except FileNotFoundError:
        wrote_header = False

    for day in daterange(start, end):
        if day in done:
            continue
        df = fetch_daily_zip_csv("aggTrades", SYMBOL, day)
        if df is None:
            print(f"{day}: not available, skipping")
            append_checkpoint(CHECKPOINT, day)
            continue
        hourly = hourly_features(df)
        hourly.to_csv(OUT, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        append_checkpoint(CHECKPOINT, day)
        print(f"{day}: {len(df):>9,} trades -> {len(hourly)} hourly rows")
        del df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    default_end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    default_start = (pd.Timestamp.utcnow() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    ap.add_argument("--start", default=default_start)
    ap.add_argument("--end", default=default_end)
    args = ap.parse_args()
    print(f"Processing aggTrades {args.start} -> {args.end} (resumable via {CHECKPOINT})")
    main(args.start, args.end)
