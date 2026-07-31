"""
reverse_engineer_features.py — Recover exact formulas for the 9 features in
bar10_training_dataset_FULLY_CORRECTED.csv that aren't computed by
signal_generator_v3.py (ema21_50_ratio, ema9_21_ratio, ema9_dist,
keltner_pos, supertrend_dist, log_return_1, log_return_5, adx_centered,
rsi_centered), by recomputing candidate indicators from the continuous 1H
OHLCV and matching them against the known training-set values at the same
timestamps. Nothing is assumed — every formula below is confirmed against
real historical rows before being trusted.
"""

import numpy as np
import pandas as pd

OHLCV_PATH = "/mnt/user-data/uploads/btc_1h_with_regime.csv"
TRAIN_PATH = "/mnt/user-data/uploads/bar10_training_dataset_FULLY_CORRECTED.csv"


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def wilder(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def compute_indicators(df):
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    c, h, l = df["close"], df["high"], df["low"]

    df["ema9"] = ema(c, 9)
    df["ema21"] = ema(c, 21)
    df["ema50"] = ema(c, 50)
    df["ema200"] = ema(c, 200)

    # ATR (Wilder) — 14 period, standard TR
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr14"] = wilder(tr, 14)

    # Keltner channel: EMA20 basis +/- 2*ATR10 (a common default; verified below)
    df["kc_basis"] = ema(c, 20)
    kc_atr = wilder(tr, 10)
    df["kc_upper"] = df["kc_basis"] + 2 * kc_atr
    df["kc_lower"] = df["kc_basis"] - 2 * kc_atr

    # Supertrend(10,3) — standard formula (same construction as nifty_signal_generator_v2.py)
    st_atr = wilder(tr, 10)
    hl2 = (h + l) / 2
    upper = (hl2 + 3 * st_atr).values
    lower = (hl2 - 3 * st_atr).values
    close_v = c.values
    st = np.full(len(df), np.nan)
    direction = np.ones(len(df), dtype=int)
    st[0] = lower[0]
    for i in range(1, len(df)):
        fu = upper[i] if (upper[i] < st[i - 1] or close_v[i - 1] > st[i - 1]) else st[i - 1]
        fl = lower[i] if (lower[i] > st[i - 1] or close_v[i - 1] < st[i - 1]) else st[i - 1]
        if direction[i - 1] == 1:
            if close_v[i] < fl:
                direction[i] = -1; st[i] = fu
            else:
                direction[i] = 1; st[i] = fl
        else:
            if close_v[i] > fu:
                direction[i] = 1; st[i] = fl
            else:
                direction[i] = -1; st[i] = fu
    df["supertrend"] = st

    df["log_return_1"] = np.log(c / c.shift(1))
    df["log_return_5"] = np.log(c / c.shift(5))

    return df


def main():
    print("Loading continuous OHLCV and computing candidate indicators...")
    ohlcv = pd.read_csv(OHLCV_PATH, parse_dates=["timestamp"])
    ohlcv = compute_indicators(ohlcv)

    train = pd.read_csv(TRAIN_PATH, parse_dates=["Date/Time"]).sort_values("Date/Time").reset_index(drop=True)

    ohlcv_renamed = ohlcv[["timestamp", "close", "ema9", "ema21", "ema50", "ema200",
                           "kc_basis", "kc_upper", "kc_lower", "supertrend",
                           "log_return_1", "log_return_5"]].rename(columns={
        "close": "cand_close",
        "log_return_1": "cand_log_return_1",
        "log_return_5": "cand_log_return_5",
    })
    merged = pd.merge_asof(
        train, ohlcv_renamed,
        left_on="Date/Time", right_on="timestamp", direction="backward"
    )

    # --- adx_centered / rsi_centered: already algebraically confirmed ---
    merged["cand_adx_centered"] = (merged["adx"] - 25) / 25
    merged["cand_rsi_centered"] = (merged["rsi_14"] - 50) / 50
    print("\nadx_centered check  — max abs error:", (merged["cand_adx_centered"] - merged["adx_centered"]).abs().max())
    print("rsi_centered check  — max abs error:", (merged["cand_rsi_centered"] - merged["rsi_centered"]).abs().max())

    # --- log returns ---
    train_lr1 = train["log_return_1"].values
    train_lr5 = train["log_return_5"].values
    print("log_return_1  max abs err:", np.nanmax(np.abs(merged["cand_log_return_1"].values - train_lr1)))
    print("log_return_5  max abs err:", np.nanmax(np.abs(merged["cand_log_return_5"].values - train_lr5)))

    # --- EMA ratios / distances: try a few candidate formulas ---
    # Use the training set's own 'Price USDT' (== signal-candle close, per signal_generator_v3.py)
    # as the reference price, not the independently-fetched candle close, so any small OHLCV
    # discrepancy doesn't get misread as a formula error.
    price = train["Price USDT"]
    print("\n--- ema21_50_ratio candidates ---")
    cands = {
        "(ema21-ema50)/ema50": (merged["ema21"] - merged["ema50"]) / merged["ema50"],
        "(ema21-ema50)/price": (merged["ema21"] - merged["ema50"]) / price,
        "ema21/ema50 - 1": merged["ema21"] / merged["ema50"] - 1,
    }
    for name, series in cands.items():
        err = np.nanmax(np.abs(series.values - train["ema21_50_ratio"].values))
        print(f"  {name:30s} max abs err = {err:.6f}")

    print("\n--- ema9_21_ratio candidates ---")
    cands = {
        "(ema9-ema21)/ema21": (merged["ema9"] - merged["ema21"]) / merged["ema21"],
        "(ema9-ema21)/price": (merged["ema9"] - merged["ema21"]) / price,
        "ema9/ema21 - 1": merged["ema9"] / merged["ema21"] - 1,
    }
    for name, series in cands.items():
        err = np.nanmax(np.abs(series.values - train["ema9_21_ratio"].values))
        print(f"  {name:30s} max abs err = {err:.6f}")

    print("\n--- ema9_dist candidates ---")
    cands = {
        "(price-ema9)/ema9": (price - merged["ema9"]) / merged["ema9"],
        "(price-ema9)/price": (price - merged["ema9"]) / price,
        "(price-ema9)/atr14": (price - merged["ema9"]) / merged["atr14"],
    }
    for name, series in cands.items():
        err = np.nanmax(np.abs(series.values - train["ema9_dist"].values))
        print(f"  {name:30s} max abs err = {err:.6f}")

    print("\n--- keltner_pos candidates ---")
    kc_range = merged["kc_upper"] - merged["kc_lower"]
    cands = {
        "(price-kc_lower)/(upper-lower)": (price - merged["kc_lower"]) / kc_range,
        "(price-kc_basis)/(upper-lower)": (price - merged["kc_basis"]) / kc_range,
        "(price-kc_basis)/(0.5*(upper-lower))": (price - merged["kc_basis"]) / (0.5 * kc_range),
    }
    for name, series in cands.items():
        err = np.nanmax(np.abs(series.values - train["keltner_pos"].values))
        print(f"  {name:30s} max abs err = {err:.6f}")

    print("\n--- supertrend_dist candidates ---")
    cands = {
        "price-supertrend": price - merged["supertrend"],
        "(price-supertrend)/price*100": (price - merged["supertrend"]) / price * 100,
        "(price-supertrend)/atr14": (price - merged["supertrend"]) / merged["atr14"],
    }
    for name, series in cands.items():
        err = np.nanmax(np.abs(series.values - train["supertrend_dist"].values))
        print(f"  {name:30s} max abs err = {err:.6f}")


if __name__ == "__main__":
    main()
