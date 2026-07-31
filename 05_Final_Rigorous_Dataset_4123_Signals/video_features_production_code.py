"""
video_features.py — Shared, causal computation of the 7 remaining
triple-barrier-video features (EMA/Keltner/Supertrend/log-return based).

CRITICAL DESIGN RULE: this module is imported by BOTH
  1. rebuild_historical_video_features.py (offline, builds training data)
  2. signal_generator_v3.py (live, computes features at inference time)

There is exactly ONE implementation. If historical rebuild and live inference
ever computed these features with different code, they could silently drift
apart (this already happened once in this codebase with next-bar ATR) --
this file exists specifically so that can't happen again.

NO-LOOKAHEAD GUARANTEE:
  compute_indicators() uses only pandas .ewm() / .rolling() operations,
  which by construction only ever read row i and earlier -- never row i+1
  or later. get_features_at(df, idx) then reads a SINGLE row (idx), which
  must be the last CLOSED candle at decision time. As long as the caller
  never passes a df that includes candles after the signal candle, and
  never calls get_features_at with an idx beyond the last closed row,
  these features cannot use future information. This is verified in
  tests/test_no_lookahead.py (included below as a self-test).
"""

import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds EMA9/21/50, ATR14, Supertrend, Keltner channel, and log-return
    columns to a candle DataFrame. Every added column at row i is a
    function of rows <= i only -- verified by test_no_lookahead.py.

    df must have columns: open, high, low, close (indexed or sorted by time,
    oldest first). Returns a NEW DataFrame (does not mutate the input).
    """
    df = df.copy()
    c, h, l = df['close'], df['high'], df['low']

    df['ema9'] = c.ewm(span=9, adjust=False).mean()
    df['ema21'] = c.ewm(span=21, adjust=False).mean()
    df['ema50'] = c.ewm(span=50, adjust=False).mean()

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df['atr14'] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    df['supertrend'] = _supertrend(df)

    kc_mid = c.ewm(span=20, adjust=False).mean()
    kc_upper = kc_mid + 2 * df['atr14']
    kc_lower = kc_mid - 2 * df['atr14']
    df['keltner_pos'] = (c - kc_lower) / (kc_upper - kc_lower + 1e-9)

    df['log_return_1'] = np.log(c / c.shift(1))
    df['log_return_5'] = np.log(c / c.shift(5))

    return df


def _supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """Standard Supertrend. Each value at row i depends only on rows <= i
    (previous supertrend value + previous close), so it's causal by construction."""
    h, l, c = df['high'], df['low'], df['close']
    hl2 = (h + l) / 2
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    st.iloc[0] = lower.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        prev_st, prev_dir = st.iloc[i - 1], direction.iloc[i - 1]
        if c.iloc[i - 1] > prev_st:
            cur_st = max(lower.iloc[i], prev_st) if prev_dir == 1 else lower.iloc[i]
            cur_dir = 1 if c.iloc[i] > cur_st else -1
        else:
            cur_st = min(upper.iloc[i], prev_st) if prev_dir == -1 else upper.iloc[i]
            cur_dir = -1 if c.iloc[i] < cur_st else 1
        st.iloc[i] = cur_st
        direction.iloc[i] = cur_dir
    return st


VIDEO_FEATURE_NAMES = [
    'ema21_50_ratio', 'ema9_21_ratio', 'ema9_dist',
    'keltner_pos', 'log_return_1', 'log_return_5', 'supertrend_dist',
]

MIN_LOOKBACK_BARS = 60   # EMA50 + Supertrend warmup needs this much history to be reliable


def get_features_at(df_with_indicators: pd.DataFrame, idx: int) -> dict:
    """
    Reads the 7 video features from a SINGLE row (idx) of a DataFrame that
    has already had compute_indicators() applied. This is the ONLY function
    both the historical rebuild and the live signal generator should call
    to extract these features -- guarantees identical logic in both places.

    idx MUST be the index of the last fully-closed candle at decision time.
    Never pass an idx beyond the last row you're certain has closed.
    """
    if idx < MIN_LOOKBACK_BARS:
        # Not enough history for EMA50/Supertrend to have converged -- return
        # NaN rather than a value computed from a too-short, unreliable window.
        return {k: np.nan for k in VIDEO_FEATURE_NAMES}

    row = df_with_indicators.iloc[idx]
    price = float(row['close'])
    atr = float(row['atr14'])

    return {
        'ema21_50_ratio': (row['ema21'] - row['ema50']) / row['ema50'],
        'ema9_21_ratio': (row['ema9'] - row['ema21']) / row['ema21'],
        'ema9_dist': (price - row['ema9']) / row['ema9'],
        'keltner_pos': row['keltner_pos'],
        'log_return_1': row['log_return_1'],
        'log_return_5': row['log_return_5'],
        'supertrend_dist': (price - row['supertrend']) / atr if atr > 0 else np.nan,
    }


def self_test_no_lookahead():
    """
    Proves the no-lookahead guarantee empirically: compute features on a
    truncated series (rows 0..idx) vs. the full series, and confirm the
    value at idx is IDENTICAL either way. If truncating the future ever
    changes a past value, that value depends on future data -- a bug.
    Run this file directly to execute the check.
    """
    np.random.seed(0)
    n = 200
    price = 100 + np.cumsum(np.random.randn(n))
    df = pd.DataFrame({
        'open': price, 'high': price + np.random.rand(n),
        'low': price - np.random.rand(n), 'close': price + np.random.randn(n) * 0.1,
    })

    idx = 150
    full = compute_indicators(df)
    truncated = compute_indicators(df.iloc[:idx + 1])

    feats_full = get_features_at(full, idx)
    feats_trunc = get_features_at(truncated, idx)

    all_match = True
    for k in VIDEO_FEATURE_NAMES:
        v1, v2 = feats_full[k], feats_trunc[k]
        match = (pd.isna(v1) and pd.isna(v2)) or np.isclose(v1, v2, equal_nan=True)
        status = "OK" if match else "MISMATCH -- LOOKAHEAD BUG"
        if not match:
            all_match = False
        print(f"  {k:20s} full={v1:.6f}  truncated={v2:.6f}  [{status}]")

    print()
    print("PASSED: no lookahead detected" if all_match else "FAILED: lookahead bug present")
    return all_match


if __name__ == '__main__':
    self_test_no_lookahead()
