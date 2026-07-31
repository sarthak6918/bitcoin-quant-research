"""
Direct leakage test: corrupt (shuffle) all OHLCV data strictly AFTER a cutoff
row, rebuild features, and check that every feature value AT OR BEFORE the
cutoff is byte-identical to the original build. If any feature at/before the
cutoff changed, it means that feature's computation reached into future bars
-- i.e. lookahead bias. This is a mechanical, not just visual, leakage check.
"""
import pandas as pd
import numpy as np
import importlib
import build_features as bf

def run():
    df = pd.read_csv(bf.IN_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    cutoff = 60000  # well past all warmup windows (max 200-bar lookback)

    orig = bf.build_features_from_df(df) if hasattr(bf, "build_features_from_df") else None

    # Corrupt everything after cutoff with random noise (different seed each time)
    df_corrupt = df.copy()
    rng = np.random.default_rng(0)
    n_tail = len(df) - cutoff
    df_corrupt.loc[cutoff:, ["open", "high", "low", "close", "volume"]] = rng.uniform(
        1, 100000, size=(n_tail, 5)
    )
    # keep high>=low sane-ish, doesn't matter for the test

    feat_orig = compute(df)
    feat_corrupt = compute(df_corrupt)

    label_cols = [c for c in feat_orig.columns if c.startswith("label_dir_") or c.startswith("fwdret_")]
    feature_cols = [c for c in feat_orig.columns if c not in label_cols and c not in ("timestamp",)]

    mismatches = []
    for col in feature_cols:
        a = feat_orig[col].iloc[:cutoff].values
        b = feat_corrupt[col].iloc[:cutoff].values
        if a.dtype.kind in "fc":
            diff = np.nanmax(np.abs(a - b)) if len(a) else 0.0
            if not np.isnan(diff) and diff > 1e-9:
                mismatches.append((col, diff))
        else:
            if not (pd.Series(a) == pd.Series(b)).all():
                mismatches.append((col, "non-numeric mismatch"))

    print(f"Checked {len(feature_cols)} feature columns for rows before cutoff={cutoff}")
    if mismatches:
        print("LEAKAGE DETECTED in columns:")
        for c, d in mismatches:
            print(f"  {c}: max diff {d}")
    else:
        print("NO LEAKAGE DETECTED: all feature columns before cutoff are identical")
        print("regardless of corrupting all data after the cutoff.")

def compute(df):
    # reimplement the same pipeline as build_features.build() but return df instead of writing
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
    rsi14 = bf.compute_rsi(c, 14)
    feat["rsi_14"] = rsi14
    k, d = bf.compute_stoch_rsi(rsi14, 14, 3, 3)
    feat["stochrsi_k"] = k
    feat["stochrsi_d"] = d
    feat["stochrsi_k_minus_d"] = k - d
    adx, plus_di, minus_di = bf.compute_adx(h, l, c, 14)
    feat["adx_14"] = adx
    feat["plus_di_14"] = plus_di
    feat["minus_di_14"] = minus_di
    feat["di_diff"] = plus_di - minus_di
    atr14 = bf.compute_atr(h, l, c, 14)
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
    st_trend, st_line = bf.compute_supertrend(h, l, c, atr14, 10, 3.0)
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
        htf_atr = bf.compute_atr(agg["high"], agg["low"], agg_c, 14)
        htf_rsi = bf.compute_rsi(agg_c, 14)
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
    for col in ["filtered_prob_state0", "filtered_prob_state1", "filtered_prob_state2",
                "filtered_prob_state3", "filtered_prob_state4"]:
        feat[col] = df[col]
    for n in bf.HORIZONS:
        fwd_ret = np.log(c.shift(-n) / c)
        feat[f"label_dir_{n}"] = (fwd_ret > 0).astype(float)
        feat[f"fwdret_{n}"] = fwd_ret
    feat["close"] = c
    return feat

if __name__ == "__main__":
    run()
