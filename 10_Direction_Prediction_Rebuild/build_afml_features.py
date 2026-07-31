"""
Add the lecture-derived features to the existing causal feature set:

  - Fractionally differentiated log price at several d (Lecture 3, slides
    24-29). The d-sweep (ffd_d_sweep.csv) shows BTC hourly log price becomes
    stationary at d=0.1 while retaining 0.979 correlation with the original
    series, whereas the log returns we have been using (d=1) retain 0.011.
    Every existing return feature is therefore memory-less by construction;
    these are the memory-preserving counterparts.

  - FFD of the same series at higher d, giving the model a multi-scale
    memory decomposition rather than one arbitrary differencing order.

  - CUSUM structural-break features (Lecture 3, slide 10): bars since the
    last symmetric-CUSUM event, and local event intensity. The threshold is
    a rolling volatility estimate, not a constant -- BTC volatility spans
    orders of magnitude over 2017-2026, so a fixed h would fire constantly
    early on and never later.

All features are causal; verified by the corrupt-the-future test in
test_afml_features.py.
"""
import numpy as np
import pandas as pd

from afml_lectures import frac_diff_ffd, cusum_filter

FFD_DS = [0.1, 0.2, 0.3, 0.4, 0.5]
FFD_THRES = 1e-4


def add_afml_features(df, feat):
    """
    df   : raw OHLCV with a `close` column, sorted ascending
    feat : existing feature frame (same row alignment) to append onto
    """
    c = df["close"].astype(float)
    logp = np.log(c)
    out = feat.copy()

    # ---- fractionally differentiated log price, multiple orders ----
    for d in FFD_DS:
        fd = frac_diff_ffd(logp, d, FFD_THRES)
        tag = str(d).replace(".", "")
        out[f"ffd_logp_d{tag}"] = fd.values
        # normalize by a trailing vol of the same series so the scale is
        # comparable across regimes (still causal: trailing window only)
        roll_sd = fd.rolling(200, min_periods=50).std()
        out[f"ffd_logp_d{tag}_z"] = (fd / roll_sd).values

    # ---- CUSUM structural-break event features ----
    logret1 = logp.diff()
    vol = logret1.rolling(100, min_periods=50).std()
    # threshold = 2 sigma of recent 1-bar returns, shifted so the threshold
    # at bar t uses only information through t-1
    h = (2.0 * vol).shift(1).values

    ev = cusum_filter(logp.values, h)
    is_event = np.zeros(len(df), dtype=float)
    if len(ev):
        is_event[ev] = 1.0
    out["cusum_event"] = is_event

    # bars since the last CUSUM event (causal by construction)
    since = np.full(len(df), np.nan)
    last = -1
    for i in range(len(df)):
        if last >= 0:
            since[i] = i - last
        if is_event[i] == 1.0:
            last = i
    out["cusum_bars_since"] = since

    # local event intensity: events in the trailing 100 bars
    out["cusum_intensity_100"] = pd.Series(is_event).rolling(100, min_periods=20).sum().values

    return out


def main():
    from build_features import IN_PATH

    raw = pd.read_csv(IN_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    feat = pd.read_csv("features_labeled.csv", parse_dates=["timestamp"])
    assert len(raw) == len(feat), (len(raw), len(feat))
    assert (raw["timestamp"].values == feat["timestamp"].values).all(), "row misalignment"

    out = add_afml_features(raw, feat)
    new_cols = [c for c in out.columns if c not in feat.columns]
    print(f"added {len(new_cols)} features: {new_cols}")
    out.to_csv("features_labeled_afml.csv", index=False)
    print(f"wrote features_labeled_afml.csv  ({len(out)} rows, {len(out.columns)} cols)")
    print(out[new_cols].isna().sum().to_string())


if __name__ == "__main__":
    main()
