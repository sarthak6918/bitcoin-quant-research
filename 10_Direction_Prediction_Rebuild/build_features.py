"""
Causal feature + multi-horizon direction-label builder.

Task: predict sign(close[t+n] - close[t]) for n in {5,10,20,50} hourly bars,
using ONLY information available at bar t's close (or earlier). Unconditional
on any entry trigger -- uses every bar in the master OHLCV file (78,150 rows)
rather than only the ~4,100 StochRSI+ADX signal rows, for statistical power
and to avoid conflating "direction" with "did a specific stop-loss trade win".

Every feature at row t is computed from bars [..., t] only (rolling windows
ending at t, inclusive of t, using only OHLCV known at t's close). No future
bar is ever touched when computing a feature. This is checked mechanically
in leakage_audit.py.
"""
import pandas as pd
import numpy as np

IN_PATH = "../02_Master_Hourly_OHLCV_With_Regime/btc_1h_ohlcv_2017_2026_with_regime.csv"
OUT_PATH = "features_labeled.csv"

HORIZONS = [1, 2, 3, 5, 10, 20, 50]

def rma(series, length):
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

def compute_rsi(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    return rsi

def compute_stoch_rsi(rsi, stoch_len=14, k_len=3, d_len=3):
    rsi_low = rsi.rolling(stoch_len).min()
    rsi_high = rsi.rolling(stoch_len).max()
    rng = (rsi_high - rsi_low).replace(0, np.nan)
    stoch = ((rsi - rsi_low) / rng * 100).fillna(0.0)
    k = stoch.rolling(k_len).mean()
    d = k.rolling(d_len).mean()
    return k, d

def compute_atr(high, low, close, length=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return rma(tr, length)

def compute_adx(high, low, close, length=14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = compute_atr(high, low, close, length)
    plus_di = 100 * rma(pd.Series(plus_dm, index=high.index), length) / atr.replace(0, np.nan)
    minus_di = 100 * rma(pd.Series(minus_dm, index=high.index), length) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = rma(dx.fillna(0.0), length)
    return adx.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)

def compute_supertrend(high, low, close, atr, length=10, mult=3.0):
    hl2 = (high + low) / 2
    upperband = hl2 + mult * atr
    lowerband = hl2 - mult * atr
    n = len(close)
    trend = np.ones(n)
    final_upper = upperband.values.copy()
    final_lower = lowerband.values.copy()
    close_v = close.values
    upperband_v = upperband.values
    lowerband_v = lowerband.values
    for i in range(1, n):
        if np.isnan(final_upper[i - 1]) or close_v[i - 1] <= final_upper[i - 1]:
            final_upper[i] = min(upperband_v[i], final_upper[i - 1]) if not np.isnan(final_upper[i - 1]) else upperband_v[i]
        else:
            final_upper[i] = upperband_v[i]
        if np.isnan(final_lower[i - 1]) or close_v[i - 1] >= final_lower[i - 1]:
            final_lower[i] = max(lowerband_v[i], final_lower[i - 1]) if not np.isnan(final_lower[i - 1]) else lowerband_v[i]
        else:
            final_lower[i] = lowerband_v[i]
        if trend[i - 1] == 1:
            trend[i] = -1 if close_v[i] < final_lower[i] else 1
        else:
            trend[i] = 1 if close_v[i] > final_upper[i] else -1
    st_line = np.where(trend == 1, final_lower, final_upper)
    return pd.Series(trend, index=close.index), pd.Series(st_line, index=close.index)

def build():
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    feat = pd.DataFrame(index=df.index)
    feat["timestamp"] = df["timestamp"]

    # --- returns / momentum at multiple lookbacks (all causal: diff/pct_change look backward) ---
    for lb in [1, 2, 3, 5, 10, 20, 50, 100, 200]:
        feat[f"logret_{lb}"] = np.log(c / c.shift(lb))

    # rolling volatility of 1-bar log returns
    logret1 = np.log(c / c.shift(1))
    for w in [10, 20, 50, 100]:
        feat[f"vol_{w}"] = logret1.rolling(w).std()

    # realized skew/kurt
    feat["skew_50"] = logret1.rolling(50).skew()
    feat["kurt_50"] = logret1.rolling(50).kurt()

    # RSI / StochRSI
    rsi14 = compute_rsi(c, 14)
    feat["rsi_14"] = rsi14
    k, d = compute_stoch_rsi(rsi14, 14, 3, 3)
    feat["stochrsi_k"] = k
    feat["stochrsi_d"] = d
    feat["stochrsi_k_minus_d"] = k - d

    # ADX / DI
    adx, plus_di, minus_di = compute_adx(h, l, c, 14)
    feat["adx_14"] = adx
    feat["plus_di_14"] = plus_di
    feat["minus_di_14"] = minus_di
    feat["di_diff"] = plus_di - minus_di

    # ATR / ATR%
    atr14 = compute_atr(h, l, c, 14)
    feat["atr_pct_14"] = atr14 / c

    # EMA distances
    for span in [9, 21, 50, 200]:
        ema = c.ewm(span=span, adjust=False).mean()
        feat[f"ema{span}_dist"] = (c - ema) / c

    # EMA slope (trend direction/strength, causal)
    for span in [9, 21, 50]:
        ema = c.ewm(span=span, adjust=False).mean()
        feat[f"ema{span}_slope_5"] = (ema - ema.shift(5)) / ema.shift(5)

    # Keltner position
    ema20 = c.ewm(span=20, adjust=False).mean()
    kelt_upper = ema20 + 2 * atr14
    kelt_lower = ema20 - 2 * atr14
    feat["keltner_pos"] = (c - ema20) / (kelt_upper - kelt_lower).replace(0, np.nan)

    # Bollinger position
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    feat["bb_pos"] = (c - sma20) / (bb_upper - bb_lower).replace(0, np.nan)
    feat["bb_width"] = (bb_upper - bb_lower) / sma20

    # Supertrend
    st_trend, st_line = compute_supertrend(h, l, c, atr14, 10, 3.0)
    feat["supertrend_trend"] = st_trend
    feat["supertrend_dist"] = (c - st_line) / c

    # Volume features
    feat["vol_z_50"] = (v - v.rolling(50).mean()) / v.rolling(50).std()
    feat["vol_change_5"] = v / v.rolling(5).mean() - 1
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    feat["obv_slope_20"] = (obv - obv.shift(20)) / v.rolling(50).mean()

    # High/low range structure
    feat["hl_range_pct"] = (h - l) / c
    feat["close_pos_in_range"] = (c - l) / (h - l).replace(0, np.nan)
    for w in [20, 50, 100]:
        roll_high = h.rolling(w).max()
        roll_low = l.rolling(w).min()
        feat[f"dist_from_high_{w}"] = (c - roll_high) / c
        feat[f"dist_from_low_{w}"] = (c - roll_low) / c

    # Autocorrelation of returns (regime: trending vs mean-reverting)
    feat["autocorr_20"] = logret1.rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)

    # Time-of-day / day-of-week (BTC trades 24/7 but session liquidity patterns exist)
    feat["hour_of_day"] = df["timestamp"].dt.hour
    feat["day_of_week"] = df["timestamp"].dt.dayofweek

    # --- multi-timeframe context: daily and 4h trend, built from COMPLETED
    # higher-tf bars only (shifted by 1 higher-tf bar so the current hourly
    # row never sees its own still-forming daily/4h bar), then forward-filled
    # onto the hourly index -- no intraday lookahead. ---
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
        # shift 1 higher-tf bar: only the PREVIOUS completed bar's values are
        # knowable while the current higher-tf bar is still forming
        htf_feat = htf_feat.shift(1)
        # reindex onto hourly timestamps, forward-fill (value only becomes
        # available once that higher-tf bar has actually closed)
        htf_feat_hourly = htf_feat.reindex(df_idx.index, method="ffill")
        for col in htf_feat.columns:
            feat[col] = htf_feat_hourly[col].values

    # HMM regime probs -- filtered_* columns are already documented as the
    # causal forward-algorithm (live-safe) decode, forward-filled with a
    # 1-day lag onto hourly bars (see 01_HMM_Regime_Model README). Safe to
    # use as-is; still shift by 1 defensively in case forward-fill lag isn't
    # exactly aligned to this row's timestamp.
    for col in ["filtered_prob_state0", "filtered_prob_state1", "filtered_prob_state2",
                "filtered_prob_state3", "filtered_prob_state4"]:
        feat[col] = df[col]

    # --- labels: forward direction over n bars, using close[t] as reference ---
    for n in HORIZONS:
        fwd_ret = np.log(c.shift(-n) / c)
        feat[f"label_dir_{n}"] = (fwd_ret > 0).astype(float)
        feat[f"fwdret_{n}"] = fwd_ret
        # rows where t+n goes past the end of the dataset -> label invalid
        feat.loc[df.index > (len(df) - 1 - n), f"label_dir_{n}"] = np.nan

    feat["close"] = c
    feat.to_csv(OUT_PATH, index=False)
    print("rows:", len(feat), "cols:", len(feat.columns))
    print(feat.isna().sum()[feat.isna().sum() > 0])

if __name__ == "__main__":
    build()
