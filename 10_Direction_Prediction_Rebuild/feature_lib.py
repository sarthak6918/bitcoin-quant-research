"""
Shared, reusable causal feature computation -- used identically by the
offline dataset builder (build_features.py, for training) and the live
inference script (live_inference.py, for production scoring), so there is
NO possibility of train/serve feature skew (a leakage/bug class this
project has already been burned by once -- see 09_Bugs_Found_And_Fixed).

Regime (HMM) probabilities are deliberately NOT included here: dropping
them cost 0.0 AUC on the frozen test (0.5397 without vs 0.5382 with, on
2025-2026 holdout) and removes a fragile live dependency (hmmlearn does
not have a prebuilt wheel in this environment). See README for the exact
comparison.
"""
import pandas as pd
import numpy as np
from build_features import (
    compute_rsi, compute_stoch_rsi, compute_atr, compute_adx, compute_supertrend
)

MIN_WARMUP_HOURS = 220  # max lookback used below (ema200) + margin


def build_live_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: raw hourly OHLCV with columns [timestamp, open, high, low, close, volume],
    sorted ascending, timestamp tz-naive UTC, NO gaps ideally.
    Returns a feature dataframe aligned 1:1 with df's rows (NaN during warmup).
    Every column here is computed using only data at or before that row --
    identical logic to build_features.py / leakage_audit.py, mechanically
    verified there via the corrupt-the-future test.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    feat = pd.DataFrame(index=df.index)
    feat["timestamp"] = df["timestamp"]

    for lb in [1, 2, 3, 5, 10, 20, 50, 100, 200]:
        feat[f"logret_{lb}"] = np.log(c / c.shift(lb))

    logret1 = np.log(c / c.shift(1))
    for w in [10, 20, 50, 100]:
        feat[f"vol_{w}"] = logret1.rolling(w).std()

    feat["skew_50"] = logret1.rolling(50).skew()
    feat["kurt_50"] = logret1.rolling(50).kurt()

    rsi14 = compute_rsi(c, 14)
    feat["rsi_14"] = rsi14
    k, d = compute_stoch_rsi(rsi14, 14, 3, 3)
    feat["stochrsi_k"] = k
    feat["stochrsi_d"] = d
    feat["stochrsi_k_minus_d"] = k - d

    adx, plus_di, minus_di = compute_adx(h, l, c, 14)
    feat["adx_14"] = adx
    feat["plus_di_14"] = plus_di
    feat["minus_di_14"] = minus_di
    feat["di_diff"] = plus_di - minus_di

    atr14 = compute_atr(h, l, c, 14)
    feat["atr_pct_14"] = atr14 / c

    for span in [9, 21, 50, 200]:
        ema = c.ewm(span=span, adjust=False).mean()
        feat[f"ema{span}_dist"] = (c - ema) / c

    for span in [9, 21, 50]:
        ema = c.ewm(span=span, adjust=False).mean()
        feat[f"ema{span}_slope_5"] = (ema - ema.shift(5)) / ema.shift(5)

    ema20 = c.ewm(span=20, adjust=False).mean()
    kelt_upper = ema20 + 2 * atr14
    kelt_lower = ema20 - 2 * atr14
    feat["keltner_pos"] = (c - ema20) / (kelt_upper - kelt_lower).replace(0, np.nan)

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    feat["bb_pos"] = (c - sma20) / (bb_upper - bb_lower).replace(0, np.nan)
    feat["bb_width"] = (bb_upper - bb_lower) / sma20

    st_trend, st_line = compute_supertrend(h, l, c, atr14, 10, 3.0)
    feat["supertrend_trend"] = st_trend
    feat["supertrend_dist"] = (c - st_line) / c

    feat["vol_z_50"] = (v - v.rolling(50).mean()) / v.rolling(50).std()
    feat["vol_change_5"] = v / v.rolling(5).mean() - 1
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    feat["obv_slope_20"] = (obv - obv.shift(20)) / v.rolling(50).mean()

    feat["hl_range_pct"] = (h - l) / c
    feat["close_pos_in_range"] = (c - l) / (h - l).replace(0, np.nan)
    for w in [20, 50, 100]:
        roll_high = h.rolling(w).max()
        roll_low = l.rolling(w).min()
        feat[f"dist_from_high_{w}"] = (c - roll_high) / c
        feat[f"dist_from_low_{w}"] = (c - roll_low) / c

    feat["autocorr_20"] = logret1.rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)

    feat["hour_of_day"] = df["timestamp"].dt.hour
    feat["day_of_week"] = df["timestamp"].dt.dayofweek

    df_idx = df.set_index("timestamp")
    for rule, tag in [("4h", "4h"), ("1D", "1d")]:
        agg = df_idx[["open", "high", "low", "close", "volume"]].resample(rule).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        agg_c = agg["close"]
        ema20_htf = agg_c.ewm(span=20, adjust=False).mean()
        htf_atr = compute_atr(agg["high"], agg["low"], agg_c, 14)
        htf_rsi = compute_rsi(agg_c, 14)
        htf_feat = pd.DataFrame({
            f"htf_{tag}_ema20_dist": (agg_c - ema20_htf) / agg_c,
            f"htf_{tag}_atr_pct": htf_atr / agg_c,
            f"htf_{tag}_rsi14": htf_rsi,
            f"htf_{tag}_logret5": np.log(agg_c / agg_c.shift(5)),
        })
        htf_feat = htf_feat.shift(1)
        htf_feat_hourly = htf_feat.reindex(df_idx.index, method="ffill")
        for col in htf_feat.columns:
            feat[col] = htf_feat_hourly[col].values

    feat["close"] = c
    return feat
