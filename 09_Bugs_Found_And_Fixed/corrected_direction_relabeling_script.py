"""
relabel_live_holdout_triple_barrier.py — Replace the bar10-close proxy label
with the ACTUAL triple-barrier label the model was trained to predict,
confirmed exact (machine-epsilon match across all 2,827 non-vertical
training rows): barrier_pct = max(2 * atr_pct_at_entry, 1.0), vertical
timeout at 10 bars, upper=favorable / lower=stop-loss.

Scans the continuous 1H OHLCV bar-by-bar (using high/low, not just close)
from entry+1 through entry+10 to find which barrier is hit first, exactly
mirroring how the historical dataset's barrier_hit/pct_profit_bar10 columns
must have been constructed.
"""

import numpy as np
import pandas as pd

OHLCV_PATH = "/mnt/user-data/uploads/btc_1h_with_regime.csv"
LIVE_PATH = "/tmp/combined_live_holdout_with_atr.csv"  # built just before this


def label_one_trade(entry_time, entry_price, direction, atr_pct, ohlcv, ts_to_pos):
    """direction: 'BUY' or 'SELL'. Returns (pct_profit, binary_target, barrier_hit, bars_to_exit)."""
    barrier_pct = max(2 * atr_pct, 1.0)
    if direction == "BUY":
        upper_price = entry_price * (1 + barrier_pct / 100)
        lower_price = entry_price * (1 - barrier_pct / 100)
    else:
        upper_price = entry_price * (1 - barrier_pct / 100)  # favorable = price falls
        lower_price = entry_price * (1 + barrier_pct / 100)  # stop = price rises

    entry_pos = ts_to_pos.get(entry_time, None)
    if entry_pos is None:
        return np.nan, np.nan, "no_data", np.nan

    for bars in range(1, 11):
        pos = entry_pos + bars
        if pos >= len(ohlcv):
            return np.nan, np.nan, "no_data", np.nan
        bar = ohlcv.iloc[pos]
        hi, lo = bar["high"], bar["low"]

        if direction == "BUY":
            hit_upper = hi >= upper_price
            hit_lower = lo <= lower_price
        else:
            hit_upper = lo <= upper_price
            hit_lower = hi >= lower_price

        if hit_upper and hit_lower:
            # both touched in the same bar -- conservative assumption: stop-loss first
            return -barrier_pct, 0, "lower_ambiguous", bars
        if hit_upper:
            return barrier_pct, 1, "upper", bars
        if hit_lower:
            return -barrier_pct, 0, "lower", bars

        if bars == 10:
            close_10 = bar["close"]
            pct = ((close_10 - entry_price) / entry_price * 100 if direction == "BUY"
                   else (entry_price - close_10) / entry_price * 100)
            return pct, int(pct > 0), "vertical", 10

    return np.nan, np.nan, "no_data", np.nan


def main():
    ohlcv = pd.read_csv(OHLCV_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts_to_pos = pd.Series(np.arange(len(ohlcv)), index=ohlcv["timestamp"].values)

    live = pd.read_csv(LIVE_PATH, parse_dates=["Date/Time"])

    results = []
    for _, row in live.iterrows():
        direction = row["execution_signal"].strip().upper()
        atr_pct = row["atr_pct"]
        pct, target, hit, bars = label_one_trade(
            row["Date/Time"], row["Price USDT"], direction, atr_pct, ohlcv, ts_to_pos
        )
        results.append((pct, target, hit, bars))

    live["pct_profit_tb"], live["binary_target_tb"], live["barrier_hit_tb"], live["bars_to_exit_tb"] = zip(*results)

    print("Barrier hit distribution:")
    print(live["barrier_hit_tb"].value_counts())
    print()
    print("Rows with no_data (insufficient forward history):", (live["barrier_hit_tb"] == "no_data").sum())
    valid = live[live["barrier_hit_tb"] != "no_data"].copy()
    print(f"\nValid relabeled rows: {len(valid)} of {len(live)}")
    print("Win rate (triple-barrier label):", valid["binary_target_tb"].mean().round(4))
    print("Win rate (old bar10-close proxy):", valid["binary_target_live"].mean().round(4))
    print("Agreement between old proxy label and correct triple-barrier label:",
          (valid["binary_target_live"] == valid["binary_target_tb"]).mean().round(4))

    valid.to_csv("/tmp/live_holdout_triple_barrier.csv", index=False)
    print("\nsaved /tmp/live_holdout_triple_barrier.csv")


if __name__ == "__main__":
    main()
