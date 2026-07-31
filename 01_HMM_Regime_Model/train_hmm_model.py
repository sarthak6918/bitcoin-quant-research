"""
Gaussian HMM regime detector for BTC/USDT daily bars. Fits K states
(bull/bear/chop, low/high vol) on log return + realized vol, picks K by
BIC against held-out log-likelihood, then decodes two ways: a causal
forward-only pass that's safe for live/backtest use, and a full
forward-backward smoother for after-the-fact regime labeling only.

Also dumps a daily->1H mapping so an intraday pipeline can pick up the
regime tag without re-fitting anything.

Usage:
    python btc_regime_hmm.py --start_date 2017-08-17 --k_min 2 --k_max 5 \
        --features logret,vol --n_restarts 20 --plot
    python btc_regime_hmm.py --map_1h /root/mlbot/some_1h_ohlcv.csv

Spot daily close only for now; funding rate/OI and any order-book data
are out of scope here.
"""

import argparse
import pickle
import sys
import time
import warnings
import logging

logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)  # suppress "Model is not converging"
                                                            # spam — these are per-restart EM
                                                            # diagnostics; we handle instability
                                                            # via explicit health checks instead.
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# CONFIG DEFAULTS
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1d"
KLINES_LIMIT = 1000  # Binance max per call

DEFAULT_START_DATE = "2017-08-17"  # first day BTCUSDT spot has data on Binance
OUTPUT_DIR = Path("./btc_regime_output")


# 1. DATA FETCH — BTC/USDT daily OHLCV, paginated back to start_date
def fetch_daily_ohlcv(start_date: str, symbol: str = SYMBOL, cache_path: Path = None) -> pd.DataFrame:
    """
    Fetch daily BTC/USDT OHLCV from Binance public REST API, paginated
    forward from start_date to now. No API key needed (public endpoint).
    """
    if cache_path is not None and cache_path.exists():
        print(f"  ✓ Using cached OHLCV: {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df

    start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    now_ts = int(pd.Timestamp.utcnow().timestamp() * 1000)

    all_rows = []
    cursor = start_ts
    print(f"  Fetching {symbol} {INTERVAL} klines from {start_date} ...")
    while cursor < now_ts:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "startTime": cursor,
            "limit": KLINES_LIMIT,
        }
        for attempt in range(5):
            try:
                r = requests.get(BINANCE_KLINES_URL, params=params, timeout=20)
                if r.status_code == 200:
                    batch = r.json()
                    break
                else:
                    print(f"    ✗ HTTP {r.status_code}: {r.text[:150]} (retry {attempt+1}/5)")
                    time.sleep(2)
            except Exception as e:
                print(f"    ✗ Request error: {e} (retry {attempt+1}/5)")
                time.sleep(2)
        else:
            raise RuntimeError("Failed to fetch klines after 5 retries")

        if not batch:
            break

        all_rows.extend(batch)
        last_open_time = batch[-1][0]
        cursor = last_open_time + 24 * 60 * 60 * 1000  # advance one day past last bar
        print(f"    ... {len(all_rows)} bars so far, last = "
              f"{pd.to_datetime(last_open_time, unit='ms').date()}")
        if len(batch) < KLINES_LIMIT:
            break
        time.sleep(0.2)  # be polite to the API

    if not all_rows:
        raise RuntimeError("No data returned from Binance — check network/start_date")

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df = df[["date", "open", "high", "low", "close", "volume"]].drop_duplicates("date").reset_index(drop=True)

    if cache_path is not None:
        df.to_csv(cache_path, index=False)
        print(f"  ✓ Saved raw OHLCV → {cache_path}")

    print(f"  ✓ Fetched {len(df)} daily bars: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# 2. FEATURE CONSTRUCTION — kept deliberately low-dimensional for the HMM
def compute_features(df: pd.DataFrame, feature_set: list, vol_window: int = 14,
                      ret_smooth: int = 3, autocorr_window: int = 10) -> pd.DataFrame:
    """
    df must have columns: date, open, high, low, close, volume (daily bars).
    feature_set: subset of {'logret', 'vol', 'adx', 'autocorr'}

    - logret     : rolling-mean log return over `ret_smooth` days (drift/direction)
    - vol        : rolling std of 1-day log returns over `vol_window` days (realized vol)
    - adx        : 14-period ADX computed on daily bars (trend strength, via `ta`)
    - autocorr   : rolling lag-1 autocorrelation of 1-day log returns over
                   `autocorr_window` days (trend persistence vs mean-reversion proxy)
    """
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["log_ret_1d"] = np.log(out["close"] / out["close"].shift(1))

    if "logret" in feature_set:
        out["feat_logret"] = out["log_ret_1d"].rolling(ret_smooth, min_periods=ret_smooth).mean()

    if "vol" in feature_set:
        out["feat_vol"] = out["log_ret_1d"].rolling(vol_window, min_periods=vol_window).std()

    if "adx" in feature_set:
        try:
            import ta
            adx_obj = ta.trend.ADXIndicator(high=out["high"], low=out["low"], close=out["close"], window=14)
            out["feat_adx"] = adx_obj.adx()
        except ImportError:
            raise ImportError("`ta` package required for adx feature: pip install ta --break-system-packages")

    if "autocorr" in feature_set:
        def _rolling_autocorr(s, window):
            return s.rolling(window).apply(
                lambda x: pd.Series(x).autocorr(lag=1) if len(x) == window else np.nan, raw=False
            )
        out["feat_autocorr"] = _rolling_autocorr(out["log_ret_1d"], autocorr_window)

    feature_cols = [f"feat_{f}" for f in feature_set]
    out = out.dropna(subset=feature_cols).reset_index(drop=True)
    return out, feature_cols


# 3. TRAIN/TEST SPLIT — chronological, no shuffling
def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    n = len(df)
    split_idx = int(n * (1 - test_frac))
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    print(f"  Train: {train['date'].min().date()} → {train['date'].max().date()}  ({len(train)} rows)")
    print(f"  Test : {test['date'].min().date()} → {test['date'].max().date()}  ({len(test)} rows)")
    return train, test


# 4. HMM FITTING — multiple K, multiple random restarts, BIC/AIC + held-out LL
def n_hmm_params(k: int, d: int, covariance_type: str = "diag") -> int:
    """Number of free parameters in a Gaussian HMM, for BIC/AIC."""
    n_startprob = k - 1
    n_transmat = k * (k - 1)
    n_means = k * d
    if covariance_type == "diag":
        n_covars = k * d
    elif covariance_type == "full":
        n_covars = k * d * (d + 1) // 2
    else:
        raise ValueError("covariance_type must be 'diag' or 'full'")
    return n_startprob + n_transmat + n_means + n_covars


def model_is_healthy(model, X: np.ndarray, min_state_frac: float = 0.01) -> tuple:
    """
    EM sometimes converges to a fit where one state basically gets
    assigned to a couple of outlier days, which pumps up log-likelihood
    without meaning anything. Reject those before they win BIC: check
    finite params, valid transmat rows, and every state actually used
    above min_state_frac.
    """
    if not np.all(np.isfinite(model.transmat_)):
        return False, "non-finite transmat_"
    if not np.all(np.isfinite(model.means_)):
        return False, "non-finite means_"
    if not np.all(np.isfinite(model.covars_)):
        return False, "non-finite covars_"
    row_sums = model.transmat_.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        return False, f"transmat_ rows don't sum to 1 (sums={row_sums.round(4).tolist()})"
    try:
        states = model.predict(X)
    except Exception as e:
        return False, f"predict() failed: {e}"
    counts = np.bincount(states, minlength=model.n_components)
    frac = counts / len(X)
    if (frac < min_state_frac).any():
        thin = [i for i, f in enumerate(frac) if f < min_state_frac]
        return False, f"state(s) {thin} occupy <{min_state_frac*100:.1f}% of frames (fracs={frac.round(4).tolist()})"
    return True, "ok"


def fit_hmm_k(X_train: np.ndarray, k: int, n_restarts: int = 10,
              covariance_type: str = "diag", random_state_base: int = 42,
              min_state_frac: float = 0.01):
    """
    Fit GaussianHMM at a fixed K over several random restarts (EM only
    finds a local optimum) and keep the best healthy one — degenerate
    fits get thrown out here so they never make it into the K comparison.
    """
    from hmmlearn.hmm import GaussianHMM

    best_model, best_ll = None, -np.inf
    n_converged, n_unhealthy, n_failed = 0, 0, 0
    unhealthy_reasons = []

    for i in range(n_restarts):
        model = GaussianHMM(
            n_components=k,
            covariance_type=covariance_type,
            n_iter=500,
            tol=1e-4,
            random_state=random_state_base + i,
            init_params="stmc",
        )
        try:
            model.fit(X_train)
        except Exception as e:
            n_failed += 1
            continue

        healthy, reason = model_is_healthy(model, X_train, min_state_frac=min_state_frac)
        if not healthy:
            n_unhealthy += 1
            unhealthy_reasons.append(reason)
            continue

        try:
            ll = model.score(X_train)
        except Exception:
            n_failed += 1
            continue

        n_converged += 1
        if ll > best_ll:
            best_ll, best_model = ll, model

    print(f"    restarts: {n_converged} healthy, {n_unhealthy} rejected as degenerate, "
          f"{n_failed} raised an exception  (of {n_restarts} total)")
    if n_unhealthy and n_converged == 0:
        # show a sample of why, so it's diagnosable rather than a silent "all failed"
        for r in unhealthy_reasons[:3]:
            print(f"      e.g. rejected: {r}")

    return best_model, best_ll


def compare_k_range(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list,
                     k_min: int, k_max: int, n_restarts: int, covariance_type: str = "diag",
                     min_state_frac: float = 0.01):
    """
    Fit K = k_min..k_max, report train LL, BIC, AIC, held-out LL for each.
    Held-out LL uses model.score() on the test sequence — this evaluates
    the FIT quality (a legitimate aggregate likelihood comparison across
    models), separate from live state DECODING which must use filtering
    only (see filtered_state_probs below).

    Only healthy fits (see model_is_healthy) are eligible to win a given K;
    if every restart at some K is degenerate, that K is skipped and flagged
    rather than silently included in the BIC comparison.
    """
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    n_train, d = X_train.shape

    results = []
    models = {}
    for k in range(k_min, k_max + 1):
        print(f"\n  Fitting K={k} ({n_restarts} restarts, covariance='{covariance_type}') ...")
        model, train_ll = fit_hmm_k(X_train, k, n_restarts=n_restarts, covariance_type=covariance_type,
                                    min_state_frac=min_state_frac)
        if model is None:
            print(f"    ✗ K={k}: no restart produced a healthy model — excluded from comparison. "
                  f"Try more --n_restarts, a smaller --k_max, or a lower --min_state_frac.")
            continue
        n_params = n_hmm_params(k, d, covariance_type)
        bic = -2 * train_ll + n_params * np.log(n_train)
        aic = -2 * train_ll + 2 * n_params
        try:
            test_ll = model.score(X_test)
        except Exception:
            test_ll = np.nan

        results.append({
            "K": k, "train_loglik": train_ll, "test_loglik": test_ll,
            "n_params": n_params, "BIC": bic, "AIC": aic,
        })
        models[k] = model
        print(f"    K={k}: train_LL={train_ll:.1f}  test_LL={test_ll:.1f}  "
              f"BIC={bic:.1f}  AIC={aic:.1f}  (n_params={n_params})")

    results_df = pd.DataFrame(results)
    return results_df, models


# 5. STATE INTERPRETATION — means/vols, transition matrix, expected duration
def label_states(model, feature_cols: list) -> dict:
    """
    Heuristic label per state based on its mean return / mean vol, purely
    for human interpretability — does not affect the model's numbers.
    Assumes 'feat_logret' present for direction; falls back to state index
    if not available.
    """
    labels = {}
    means = model.means_
    if "feat_logret" not in feature_cols:
        return {i: f"state_{i}" for i in range(model.n_components)}

    ret_idx = feature_cols.index("feat_logret")
    vol_idx = feature_cols.index("feat_vol") if "feat_vol" in feature_cols else None

    rets = means[:, ret_idx]
    vols = means[:, vol_idx] if vol_idx is not None else np.zeros(model.n_components)

    ret_median = np.median(rets)
    vol_median = np.median(vols) if vol_idx is not None else 0

    for i in range(model.n_components):
        direction = "bull" if rets[i] > ret_median else "bear" if rets[i] < ret_median else "flat"
        vol_tag = "high-vol" if vols[i] > vol_median else "low-vol"
        # near-zero return + low vol reads better as "chop" than bull/bear
        if abs(rets[i]) < 0.15 * np.std(rets) + 1e-9:
            direction = "chop"
        labels[i] = f"{direction}_{vol_tag}"
    return labels


def report_model(model, feature_cols: list, k: int) -> str:
    lines = []
    lines.append(f"\n{'='*70}\nMODEL REPORT — K={k}\n{'='*70}")
    state_labels = label_states(model, feature_cols)

    lines.append("\nPer-state emission parameters (mean, std) per feature:")
    for i in range(k):
        mean_str = ", ".join(f"{fc}={model.means_[i, j]:.5f}" for j, fc in enumerate(feature_cols))
        # covars_ is always a full (k, d, d) matrix in hmmlearn, regardless
        # of covariance_type — diagonal entries are the per-feature variance.
        std_str = ", ".join(f"{fc}_std={np.sqrt(model.covars_[i][j, j]):.5f}" for j, fc in enumerate(feature_cols))
        lines.append(f"  State {i} [{state_labels[i]}]:  {mean_str}  |  {std_str}")

    lines.append("\nTransition matrix (rows = from-state, cols = to-state):")
    header = "        " + "  ".join(f"S{j}" for j in range(k))
    lines.append(header)
    for i in range(k):
        row = "  ".join(f"{model.transmat_[i, j]:.3f}" for j in range(k))
        lines.append(f"  S{i} [{state_labels[i]:>16}]:  {row}")

    lines.append("\nExpected state duration (1 / (1 - p_ii)), in bars (days):")
    for i in range(k):
        p_ii = model.transmat_[i, i]
        dur = 1.0 / (1.0 - p_ii) if p_ii < 1.0 else np.inf
        lines.append(f"  State {i} [{state_labels[i]}]: {dur:.1f} days  (self-transition p={p_ii:.3f})")

    lines.append("\nStationary distribution (long-run % of time in each state):")
    try:
        eigvals, eigvecs = np.linalg.eig(model.transmat_.T)
        stat_idx = np.argmin(np.abs(eigvals - 1.0))
        stat_dist = np.real(eigvecs[:, stat_idx])
        stat_dist = stat_dist / stat_dist.sum()
        for i in range(k):
            lines.append(f"  State {i} [{state_labels[i]}]: {stat_dist[i]*100:.1f}%")
    except Exception as e:
        lines.append(f"  (could not compute stationary distribution: {e})")

    return "\n".join(lines)


# 6. DECODING — FILTERED (causal, live-safe) vs SMOOTHED (historical only)
def filtered_state_probs(model, X: np.ndarray) -> np.ndarray:
    """
    P(state_t | obs_1..t) via the scaled forward algorithm — only looks
    at past/current bars, unlike hmmlearn's predict_proba which smooths
    over the whole sequence. This is the one that's safe to use live.
    """
    from scipy.stats import multivariate_normal

    n, d = X.shape
    k = model.n_components
    log_startprob = np.log(model.startprob_ + 1e-300)
    log_transmat = np.log(model.transmat_ + 1e-300)

    # Precompute emission log-likelihoods per state.
    # NOTE: hmmlearn stores covars_ as full (k, d, d) matrices regardless of
    # covariance_type ('diag' just means off-diagonals are constrained to 0
    # during fitting) — so we can use model.covars_[i] directly either way.
    covars = [model.covars_[i] for i in range(k)]
    dists = [multivariate_normal(mean=model.means_[i], cov=covars[i], allow_singular=True) for i in range(k)]
    log_emit = np.column_stack([dists[i].logpdf(X) for i in range(k)])  # (n, k)

    filtered = np.zeros((n, k))
    # t = 0
    log_alpha = log_startprob + log_emit[0]
    log_alpha -= log_alpha.max()
    alpha = np.exp(log_alpha)
    alpha /= alpha.sum()
    filtered[0] = alpha

    for t in range(1, n):
        # alpha_t(j) ∝ emit_t(j) * sum_i alpha_{t-1}(i) * transmat(i,j)
        pred = filtered[t - 1] @ model.transmat_  # (k,) predicted state dist before seeing obs t
        pred = np.clip(pred, 1e-300, None)
        unnorm = pred * np.exp(log_emit[t] - log_emit[t].max())
        s = unnorm.sum()
        filtered[t] = unnorm / s if s > 0 else pred / pred.sum()

    return filtered  # (n, k), each row sums to 1, filtered[t] uses only X[0..t]


def smoothed_state_probs(model, X: np.ndarray) -> np.ndarray:
    """
    Full forward-backward smoothed probabilities — uses the WHOLE sequence
    (past AND future). Valid for historical/economic interpretation of
    "what regime was this actually" with hindsight. NOT valid for live use.
    """
    return model.predict_proba(X)


# 7. MAP DAILY REGIME ONTO EXISTING 1H OHLCV (forward-fill)
def map_regime_to_1h(daily_regime_df: pd.DataFrame, path_1h: str, out_path: Path):
    """
    daily_regime_df must have columns: date, filtered_state, filtered_state_label,
    and one filtered_prob_state_i column per state.
    path_1h: existing 1H OHLCV csv on the VPS. Must contain a timestamp column
    named 'timestamp', 'Date/Time', or 'date' — auto-detected.
    Regime is forward-filled: each 1H bar gets the daily regime of the most
    recent COMPLETED daily bar strictly before it (no lookahead — today's
    regime state is not known until the daily bar closes).
    """
    df_1h = pd.read_csv(path_1h)
    ts_col = next((c for c in ["timestamp", "Date/Time", "date", "datetime"] if c in df_1h.columns), None)
    if ts_col is None:
        raise ValueError(f"Could not find a timestamp column in {path_1h}. "
                          f"Columns present: {list(df_1h.columns)}")
    df_1h[ts_col] = pd.to_datetime(df_1h[ts_col])

    daily = daily_regime_df.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    # regime for day D becomes known/usable only once day D has closed,
    # i.e. from day D+1 00:00 onward — shift the "available from" timestamp
    daily["available_from"] = daily["date"] + pd.Timedelta(days=1)

    merge_cols = ["available_from", "filtered_state", "filtered_state_label"] + \
                 [c for c in daily.columns if c.startswith("filtered_prob_state")]
    daily_for_merge = daily[merge_cols].rename(columns={"available_from": ts_col}).sort_values(ts_col)

    df_1h = df_1h.sort_values(ts_col)
    merged = pd.merge_asof(df_1h, daily_for_merge, on=ts_col, direction="backward")
    merged.to_csv(out_path, index=False)
    print(f"  ✓ 1H data with mapped regime → {out_path}  ({len(merged)} rows)")
    return merged


# 8. PLOT
def plot_regimes(df: pd.DataFrame, state_col: str, state_labels: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 7))
    k = len(state_labels)
    cmap = plt.get_cmap("tab10")

    ax.plot(df["date"], df["close"], color="black", linewidth=0.8, zorder=3)
    ax.set_yscale("log")

    for state in range(k):
        mask = df[state_col] == state
        if mask.sum() == 0:
            continue
        ax.scatter(df.loc[mask, "date"], df.loc[mask, "close"],
                   s=6, color=cmap(state % 10), label=f"S{state}: {state_labels[state]}", zorder=4)

    ax.set_title("BTC/USDT Daily Close — Filtered HMM Regime (causal, no lookahead)")
    ax.set_ylabel("Price (log scale)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  ✓ Plot saved → {out_path}")


# MAIN
def main():
    ap = argparse.ArgumentParser(description="BTC Market Regime HMM")
    ap.add_argument("--start_date", default=DEFAULT_START_DATE,
                     help="Earliest date to fetch daily OHLCV from (default: BTCUSDT spot inception)")
    ap.add_argument("--k_min", type=int, default=2)
    ap.add_argument("--k_max", type=int, default=5)
    ap.add_argument("--n_restarts", type=int, default=15,
                     help="Random EM restarts per K (EM only finds a local optimum)")
    ap.add_argument("--test_frac", type=float, default=0.2,
                     help="Fraction of chronological tail held out for test log-likelihood")
    ap.add_argument("--features", default="logret,vol",
                     help="Comma list from {logret,vol,adx,autocorr}. Default: logret,vol")
    ap.add_argument("--covariance_type", default="diag", choices=["diag", "full"])
    ap.add_argument("--vol_window", type=int, default=14)
    ap.add_argument("--ret_smooth", type=int, default=3)
    ap.add_argument("--autocorr_window", type=int, default=10)
    ap.add_argument("--min_state_frac", type=float, default=0.01,
                     help="Reject a fitted model if any state occupies less than this "
                          "fraction of training frames (guards against degenerate/empty "
                          "states winning on log-likelihood). Default 0.01 (1%%).")
    ap.add_argument("--select_k", type=int, default=None,
                     help="Force a specific K for the final saved model instead of auto (lowest BIC)")
    ap.add_argument("--output_dir", default=str(OUTPUT_DIR))
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--map_1h", default=None,
                     help="Path to an existing 1H OHLCV csv to map the daily regime onto")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_set = [f.strip() for f in args.features.split(",") if f.strip()]

    print("=" * 70)
    print("BTC REGIME HMM PIPELINE")
    print(f"Features: {feature_set}  |  K range: {args.k_min}-{args.k_max}  |  "
          f"restarts/K: {args.n_restarts}")
    print("=" * 70)

    # 1. Fetch data
    print("\n[1/7] Fetching daily OHLCV ...")
    raw = fetch_daily_ohlcv(args.start_date, cache_path=out_dir / "btc_daily_ohlcv.csv")

    # 2. Features
    print("\n[2/7] Computing features ...")
    feat_df, feature_cols = compute_features(
        raw, feature_set, vol_window=args.vol_window,
        ret_smooth=args.ret_smooth, autocorr_window=args.autocorr_window
    )
    print(f"  ✓ {len(feat_df)} rows after feature warmup  |  columns: {feature_cols}")

    # 3. Split
    print("\n[3/7] Chronological train/test split ...")
    train_df, test_df = chronological_split(feat_df, test_frac=args.test_frac)

    # 4. Fit K range
    print("\n[4/7] Fitting HMM across K range ...")
    results_df, models = compare_k_range(
        train_df, test_df, feature_cols, args.k_min, args.k_max,
        n_restarts=args.n_restarts, covariance_type=args.covariance_type,
        min_state_frac=args.min_state_frac
    )
    if results_df.empty:
        print("✗ No models converged to a healthy fit at any K. Try more --n_restarts, "
              "a narrower --k_min/--k_max, or a lower --min_state_frac.")
        sys.exit(1)

    print("\n  --- Model comparison ---")
    print(results_df.to_string(index=False))

    best_bic_k = int(results_df.loc[results_df["BIC"].idxmin(), "K"])
    best_test_ll_k = int(results_df.loc[results_df["test_loglik"].idxmax(), "K"])
    print(f"\n  Lowest BIC     → K={best_bic_k}")
    print(f"  Best held-out LL → K={best_test_ll_k}")
    if best_bic_k != best_test_ll_k:
        print("  (BIC and held-out LL disagree — inspect both models' state reports "
              "below before committing to one K.)")

    chosen_k = args.select_k if args.select_k is not None else best_bic_k
    chosen_model = models[chosen_k]
    print(f"\n  → Using K={chosen_k} for final report/decoding "
          f"({'user-selected' if args.select_k else 'lowest BIC'})")

    # 5. Report all fitted models (full detail for chosen K + summary for rest)
    print("\n[5/7] Building model report ...")
    report_lines = ["BTC REGIME HMM — MODEL REPORT", "=" * 70,
                     f"Data: {raw['date'].min().date()} → {raw['date'].max().date()}  "
                     f"({len(raw)} raw daily bars)",
                     f"Features used: {feature_cols}",
                     f"Train: {train_df['date'].min().date()} → {train_df['date'].max().date()} "
                     f"({len(train_df)} rows)",
                     f"Test : {test_df['date'].min().date()} → {test_df['date'].max().date()} "
                     f"({len(test_df)} rows)",
                     "\nModel comparison across K:",
                     results_df.to_string(index=False),
                     f"\nSelected K = {chosen_k} "
                     f"({'user override' if args.select_k else 'lowest BIC'}; "
                     f"best held-out LL was K={best_test_ll_k})"]
    for k, model in models.items():
        report_lines.append(report_model(model, feature_cols, k))
    report_text = "\n".join(report_lines)
    (out_dir / "btc_regime_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\n  ✓ Full report saved → {out_dir / 'btc_regime_report.txt'}")

    # 6. Decode full history with chosen model — filtered (causal) + smoothed (hindsight)
    print("\n[6/7] Decoding states (filtered=causal/live-safe, smoothed=hindsight-only) ...")
    X_full = feat_df[feature_cols].values
    filtered_probs = filtered_state_probs(chosen_model, X_full)
    smoothed_probs = smoothed_state_probs(chosen_model, X_full)
    state_labels = label_states(chosen_model, feature_cols)

    decoded = feat_df.copy()
    decoded["filtered_state"] = filtered_probs.argmax(axis=1)
    decoded["filtered_state_label"] = decoded["filtered_state"].map(state_labels)
    for i in range(chosen_k):
        decoded[f"filtered_prob_state{i}"] = filtered_probs[:, i]

    decoded["smoothed_state"] = smoothed_probs.argmax(axis=1)
    decoded["smoothed_state_label"] = decoded["smoothed_state"].map(state_labels)
    for i in range(chosen_k):
        decoded[f"smoothed_prob_state{i}"] = smoothed_probs[:, i]

    agree_pct = (decoded["filtered_state"] == decoded["smoothed_state"]).mean() * 100
    print(f"  Filtered vs smoothed state agreement: {agree_pct:.1f}% "
          f"(mismatches are mostly near regime-change boundaries — expected)")

    decoded.to_csv(out_dir / "btc_regime_features.csv", index=False)
    with open(out_dir / f"btc_regime_model_K{chosen_k}.pkl", "wb") as f:
        pickle.dump({
            "model": chosen_model,
            "feature_cols": feature_cols,
            "feature_set": feature_set,
            "state_labels": state_labels,
            "vol_window": args.vol_window,
            "ret_smooth": args.ret_smooth,
            "autocorr_window": args.autocorr_window,
            "covariance_type": args.covariance_type,
            "chosen_k": chosen_k,
        }, f)
    print(f"  ✓ Decoded states → {out_dir / 'btc_regime_features.csv'}")
    print(f"  ✓ Fitted model saved → {out_dir / f'btc_regime_model_K{chosen_k}.pkl'}")

    if args.plot:
        plot_regimes(decoded, "filtered_state", state_labels, out_dir / "btc_regime_plot.png")

    # 7. Optional: map onto existing 1H OHLCV
    if args.map_1h:
        print("\n[7/7] Mapping daily regime onto 1H OHLCV ...")
        map_regime_to_1h(decoded, args.map_1h, out_dir / "btc_1h_with_regime.csv")
    else:
        print("\n[7/7] Skipped — no --map_1h path supplied.")

    print("\n" + "=" * 70)
    print("DONE.")
    print(f"All outputs in: {out_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
