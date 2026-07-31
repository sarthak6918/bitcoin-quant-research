"""
Implementations from Marcos Lopez de Prado's ORIE 5256 lecture notes
(the PDFs in the working folder), covering the parts NOT already built in
afml.py (which covers Lecture 4's purged CV and Ch.4 sample weights).

Lecture 3 (ssrn-3257419) -- Financial Data Structures / Labeling / Weights /
                            Fractionally Differentiated Features
  - dollar_bars / volume_bars   : slides 5-6, "sample as a subordinated
                                  process of the amount of information
                                  exchanged"; dollar bars have the most
                                  stable sampling frequency
  - cusum_filter                : slide 10, getTEvents -- sample features at
                                  irregular frequencies, on structural breaks
  - get_weights_ffd / frac_diff_ffd : slides 24-29, the stationarity-vs-memory
                                  dilemma. Returns are stationary but
                                  memory-less; prices have memory but are
                                  non-stationary. FFD finds the MINIMUM
                                  differentiation that achieves stationarity
                                  while preserving maximum memory.
  - min_ffd_d                   : slide 27-28, sweep d and pick the smallest
                                  value whose ADF stat clears the 95%
                                  critical value (-2.8623 in the lecture's
                                  E-mini example, where d ~ 0.35 and
                                  correlation with the original series is
                                  still 0.995)

Lecture 6 (ssrn-3261943) -- Backtest Statistics / Multiple Testing
  - deflated_sharpe_ratio       : corrects an observed Sharpe for the number
                                  of trials, non-normality (skew/kurtosis)
                                  and sample length
  - expected_max_sharpe         : E[max SR] under the null of zero true skill,
                                  given N independent trials
  - prob_backtest_overfitting   : CSCV-based PBO (Bailey et al.)

Every function here is an original implementation written against the
algorithm as described in the lectures, and is unit-tested in
test_afml_lectures.py.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


# ============================================================================
# LECTURE 3 -- Financial Data Structures: dollar / volume bars
# ============================================================================
def _aggregate_bars(df, cum_col, threshold):
    """
    Generic information-driven bar sampler. Walks the input series in order
    and closes a bar every time the cumulative quantity (`cum_col`) crosses
    `threshold`, then resets the accumulator.

    Input df must have columns [timestamp, open, high, low, close, volume]
    and any extra column named by cum_col.
    """
    ts = df["timestamp"].values
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    v = df["volume"].values
    cum_v = df[cum_col].values

    bars = []
    start = 0
    acc = 0.0
    for i in range(len(df)):
        acc += cum_v[i]
        if acc >= threshold:
            bars.append((
                ts[i],                      # bar closes at this timestamp
                o[start],
                h[start:i + 1].max(),
                l[start:i + 1].min(),
                c[i],
                v[start:i + 1].sum(),
                i - start + 1,              # how many source bars consumed
            ))
            start = i + 1
            acc = 0.0
    return pd.DataFrame(bars, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "n_source_bars"
    ])


def dollar_bars(df, threshold):
    """
    Lecture 3, slides 5-6. Sample a new bar every time `threshold` dollars of
    notional have traded. Dollar bars exhibit more stable sampling frequency
    than tick or volume bars, and their volatility is much closer to constant
    (homoscedastic) -- which slide 13 identifies as a prerequisite for sane
    fixed-horizon labeling.
    """
    d = df.copy()
    d["dollar"] = d["close"] * d["volume"]
    return _aggregate_bars(d, "dollar", threshold)


def volume_bars(df, threshold):
    """Lecture 3, slide 5. One bar per `threshold` units of volume."""
    d = df.copy()
    return _aggregate_bars(d, "volume", threshold)


# ============================================================================
# LECTURE 3 -- CUSUM filter for event sampling (slide 10, getTEvents)
# ============================================================================
def cusum_filter(raw, h):
    """
    Symmetric CUSUM filter. Flags the index positions where a run-up or
    run-down of cumulative changes exceeds the threshold h, then resets.

    This is the lecture's `getTEvents`, translated to positional indices.
    Purpose (slide 10): "sample the features at irregular frequencies, to
    train the algorithm to predict the outcomes of specific events" -- i.e.
    downsample to rows where something structurally happened, rather than
    feeding the model every single time bar.

    Parameters
    ----------
    raw : array-like -- the series to monitor (typically log prices)
    h   : float or array-like -- threshold. If array-like, it is a per-bar
          threshold (e.g. a rolling volatility estimate), which is the
          practical variant since a fixed h is inappropriate when volatility
          changes by orders of magnitude across a decade of BTC data.

    Returns
    -------
    np.ndarray of integer positions of the sampled events.
    """
    raw = np.asarray(raw, dtype=float)
    n = len(raw)
    if np.isscalar(h):
        h_arr = np.full(n, float(h))
    else:
        h_arr = np.asarray(h, dtype=float)

    events = []
    s_pos, s_neg = 0.0, 0.0
    diff = np.diff(raw, prepend=raw[0])
    for i in range(1, n):
        d = diff[i]
        if not np.isfinite(d):
            continue
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)
        hi = h_arr[i]
        if not np.isfinite(hi) or hi <= 0:
            continue
        if s_neg < -hi:
            s_neg = 0.0
            events.append(i)
        elif s_pos > hi:
            s_pos = 0.0
            events.append(i)
    return np.array(events, dtype=int)


# ============================================================================
# LECTURE 3 -- Fractionally Differentiated Features (slides 24-29)
# ============================================================================
def get_weights_ffd(d, thres=1e-5, max_size=10000):
    """
    Binomial-series weights for fractional differentiation of order d,
    truncated at the point where |w_k| < thres (the fixed-width window
    variant, AFML sec 5.5 / lecture slides 24-29).

    w_0 = 1;  w_k = -w_{k-1} * (d - k + 1) / k
    """
    w = [1.0]
    k = 1
    while k < max_size:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    # reverse so that the LAST element aligns with the most recent obs
    return np.array(w[::-1])


def frac_diff_ffd(series, d, thres=1e-5):
    """
    Fixed-width-window fractional differentiation.

    Resolves the stationarity-vs-memory dilemma of slide 25: standard
    returns (d=1) are stationary but erase all memory, while raw prices
    (d=0) keep full memory but are non-stationary. A fractional d in
    between keeps most of the memory AND passes stationarity tests.

    CAUSALITY NOTE: the weight vector is applied to a trailing window
    ending at the current observation, so the value at t uses only
    observations <= t. This is verified by the corrupt-the-future test in
    test_afml_lectures.py.
    """
    s = pd.Series(series).astype(float)
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    if width >= len(s):
        return pd.Series(np.nan, index=s.index)

    vals = s.values
    out = np.full(len(s), np.nan)

    if not np.isnan(vals).any():
        # Vectorized: out[i] = sum_k w[k]*vals[i-width+k] is exactly the
        # convolution of vals with w reversed. ~1000x faster than the loop
        # on 78k rows, and asserted identical to it in the unit tests.
        conv = np.convolve(vals, w[::-1])[:len(vals)]
        out[width:] = conv[width:]
    else:
        for i in range(width, len(s)):
            window = vals[i - width:i + 1]
            if np.isnan(window).any():
                continue
            out[i] = np.dot(w, window)
    return pd.Series(out, index=s.index)


def min_ffd_d(series, d_range=None, thres=1e-5, conf=0.05, max_points=20000):
    """
    Lecture 3 slides 27-28: sweep d, compute the ADF statistic on the
    fractionally differentiated series, and return the smallest d whose ADF
    stat clears the critical value -- i.e. the minimum differentiation that
    achieves stationarity, preserving maximum memory.

    Returns a DataFrame (one row per d: adf stat, p-value, critical value,
    correlation with the original series) plus the selected d.
    """
    from statsmodels.tsa.stattools import adfuller

    if d_range is None:
        d_range = np.linspace(0, 1, 11)

    s = pd.Series(series).astype(float).dropna()
    if len(s) > max_points:  # ADF on 78k points is slow and unnecessary
        s = s.iloc[-max_points:]

    rows = []
    chosen = None
    for d in d_range:
        fd = frac_diff_ffd(s, d, thres).dropna()
        if len(fd) < 100:
            continue
        # with autolag=None adfuller returns 5 values (no icbest), 6 otherwise
        res = adfuller(fd.values, maxlag=1, regression="c", autolag=None)
        stat, pval, crits = res[0], res[1], res[4]
        corr = np.corrcoef(s.loc[fd.index].values, fd.values)[0, 1]
        passes = stat < crits["5%"]
        rows.append(dict(d=round(float(d), 4), adf_stat=stat, p_value=pval,
                         crit_5pct=crits["5%"], corr_with_original=corr,
                         stationary=passes, n_obs=len(fd)))
        if passes and chosen is None:
            chosen = float(d)
    return pd.DataFrame(rows), chosen


# ============================================================================
# LECTURE 6 -- Deflated Sharpe Ratio & backtest overfitting
# ============================================================================
EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials, var_sharpe=1.0):
    """
    E[max SR] across `n_trials` independent strategies whose TRUE Sharpe is
    zero. Under the null, the maximum of N draws grows like
        sqrt(var) * ( (1-g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N*e)) )
    where g is the Euler-Mascheroni constant.

    This is the benchmark an observed Sharpe must beat: searching 129,600
    configurations guarantees a high best-Sharpe even with zero real skill.
    """
    if n_trials < 2:
        return 0.0
    g = EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return np.sqrt(var_sharpe) * ((1 - g) * z1 + g * z2)


def deflated_sharpe_ratio(observed_sr, n_trials, n_obs, skew=0.0, kurtosis=3.0,
                          var_sharpe=1.0):
    """
    Deflated Sharpe Ratio (Lecture 6). Probability that the observed Sharpe
    is genuinely greater than zero, AFTER correcting for:
      - the number of trials run (multiple-testing / selection bias)
      - non-normality of returns (skew and kurtosis)
      - the length of the sample

    Returns (dsr, sr_benchmark). dsr > 0.95 is the usual bar for calling a
    result significant. All Sharpes here are per-observation (not annualized)
    and must be on the same scale as `var_sharpe`.
    """
    sr0 = expected_max_sharpe(n_trials, var_sharpe)
    if n_obs < 2:
        return np.nan, sr0
    # standard error of the Sharpe estimator under non-normality
    denom = np.sqrt(1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr ** 2)
    if not np.isfinite(denom) or denom <= 0:
        return np.nan, sr0
    z = (observed_sr - sr0) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z)), float(sr0)


def prob_backtest_overfitting(perf_matrix, n_splits=16):
    """
    Probability of Backtest Overfitting via Combinatorially Symmetric
    Cross-Validation (Bailey, Borwein, Lopez de Prado, Zhu; Lecture 6).

    perf_matrix : 2-D array, shape (n_observations, n_strategies) of
                  per-period performance (e.g. per-trade or per-bar returns)
                  for every configuration tried.

    Method: split the observation axis into `n_splits` contiguous chunks;
    for every balanced combination, designate half the chunks IS and half
    OOS; pick the best strategy by IS Sharpe; record its OOS rank. PBO is
    the fraction of combinations where the IS-best strategy lands in the
    BOTTOM half out of sample.

    PBO near 0.5 means selecting on in-sample performance is worthless --
    the IS winner is a coin flip out of sample.
    """
    from itertools import combinations

    M = np.asarray(perf_matrix, dtype=float)
    n_obs, n_strat = M.shape
    if n_strat < 2:
        return np.nan
    chunk_bounds = np.array_split(np.arange(n_obs), n_splits)

    def sharpe(x):
        sd = np.nanstd(x, axis=0)
        sd = np.where(sd == 0, np.nan, sd)
        return np.nanmean(x, axis=0) / sd

    half = n_splits // 2
    logits = []
    for is_idx in combinations(range(n_splits), half):
        oos_idx = [k for k in range(n_splits) if k not in is_idx]
        is_rows = np.concatenate([chunk_bounds[k] for k in is_idx])
        oos_rows = np.concatenate([chunk_bounds[k] for k in oos_idx])

        sr_is = sharpe(M[is_rows])
        sr_oos = sharpe(M[oos_rows])
        if np.all(np.isnan(sr_is)) or np.all(np.isnan(sr_oos)):
            continue
        best = int(np.nanargmax(sr_is))
        # relative rank of the IS-best strategy, out of sample, in (0,1)
        finite = np.isfinite(sr_oos)
        if finite.sum() < 2:
            continue
        rank = (np.sum(sr_oos[finite] < sr_oos[best]) + 1) / (finite.sum() + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))

    if not logits:
        return np.nan
    logits = np.array(logits)
    return float(np.mean(logits <= 0))  # fraction landing in the bottom half
