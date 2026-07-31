"""
AFML Ch.4 sample weighting applied to the TRIPLE-BARRIER dataset (folder 05).

This is the one place in the project where uniqueness weighting can actually
do something: folder 10's fixed-horizon labels all span exactly n bars, so
uniqueness is identical for every row and the weights collapse to uniform.
The triple-barrier labels here have VARIABLE spans -- `bars_to_exit` ranges
1..5, because a trade that hits a barrier at bar 2 stops accruing outcome
while one that runs to the vertical barrier spans all 5.

Label lifespan is taken directly from the dataset's own bookkeeping:
    t0 = ohlcv_pos                       (bar index of the signal)
    t1 = ohlcv_pos + bars_to_exit        (bar index where the label resolves)

Weighting schemes compared (everything else held identical -- same features,
same CatBoost hyperparameters, same expanding walk-forward, same seeds, as
06_Walkforward_Validation_And_Final_Model/walkforward_5bar_regime.py):

    A. none                      -- uniform weights (control)
    B. time-decay only           -- the CURRENT PRODUCTION scheme, 180-day half-life
    C. uniqueness only           -- Ch.4 sec 4.4
    D. uniqueness x time-decay   -- Ch.4 sec 4.4 + 4.7
    E. return-attribution x decay-- Ch.4 sec 4.6 + 4.7

Also adds PURGING to the walk-forward folds (Ch.7), which the original
pipeline lacked -- though with a median 13-bar gap between signals and a
max 5-bar label span, only a handful of boundary rows are affected.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from afml import num_co_events, get_avg_uniqueness, sample_weights_by_return

DATA = "../05_Final_Rigorous_Dataset_4123_Signals/master_dataset_4123_signals_2017_2026.csv"
OHLCV = "../02_Master_Hourly_OHLCV_With_Regime/btc_1h_ohlcv_2017_2026_with_regime.csv"

BASE_FEATURES = [
    "Type_encoded", "macd_histogram", "macd_signal", "rvi", "rvi_signal",
    "rsi_14", "rsi_lag1", "mfi_14", "adx", "plus_di", "minus_di", "di_net",
    "er_10", "volume", "bar_body_ratio", "adx_regime", "vol_regime",
    "er_adx_product", "adx_centered", "rsi_centered", "ema21_50_ratio",
    "ema9_21_ratio", "ema9_dist", "keltner_pos", "log_return_1",
    "log_return_5", "supertrend_dist", "atr_pct",
]
REGIME_FEATURES = [f"filtered_prob_state{i}" for i in range(5)]
ALL_FEATURES = BASE_FEATURES + REGIME_FEATURES
TARGET = "binary_target"
HALF_LIFE_DAYS = 180
N_WF_FOLDS = 5
MIN_TRAIN_FRAC = 0.40
SEEDS = [42, 43, 44, 45, 46]


def time_decay(dates, half_life_days=HALF_LIFE_DAYS):
    """The production scheme: exponential recency decay, 180-day half-life."""
    days_old = (dates.max() - dates).dt.days.values.astype(float)
    w = np.exp(-np.log(2) * days_old / half_life_days)
    return w / w.mean()


def recompute_vol_regime(df, fit_mask, atr_col="atr_pct"):
    bins = np.quantile(df.loc[fit_mask, atr_col].values, [0.25, 0.50, 0.75])
    return np.digitize(df[atr_col].values, bins=bins)


def build_weights(df, idx, n_bars, bar_logret):
    """
    Compute all five weighting schemes for the rows selected by `idx`
    (positional indices into df). Weights are always computed from the
    TRAINING rows only -- concurrency is measured among the training labels,
    never using test-period labels.
    """
    sub = df.iloc[idx]
    t0 = sub["ohlcv_pos"].values.astype(int)
    t1 = np.minimum(t0 + sub["bars_to_exit"].values.astype(int), n_bars - 1)

    avg_u = get_avg_uniqueness(n_bars, t0, t1)
    decay = time_decay(sub["Date/Time"])
    ret_w = sample_weights_by_return(n_bars, t0, t1, bar_logret)

    def norm(w):
        w = np.asarray(w, dtype=float)
        return w / w.mean() if w.mean() > 0 else np.ones_like(w)

    return {
        "A. none": np.ones(len(sub)),
        "B. time-decay only (production)": norm(decay),
        "C. uniqueness only": norm(avg_u),
        "D. uniqueness x time-decay": norm(avg_u * decay),
        "E. return-attrib x time-decay": norm(ret_w * decay),
    }, avg_u


def fit_eval(tr, y_tr, w_tr, vl, y_vl, te, y_te, seed):
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    clf = CatBoostClassifier(
        iterations=2000, learning_rate=0.02, depth=5, l2_leaf_reg=5,
        min_data_in_leaf=20, random_strength=1.5, bagging_temperature=0.8,
        loss_function="Logloss", eval_metric="AUC",
        class_weights={0: 1.0, 1: scale_pos},
        random_seed=seed, od_type="Iter", od_wait=150,
        verbose=0, allow_writing_files=False,
    )
    clf.fit(tr[ALL_FEATURES], y_tr, sample_weight=w_tr,
            eval_set=(vl[ALL_FEATURES], y_vl), use_best_model=True)
    return clf.predict_proba(te[ALL_FEATURES])[:, 1]


def main():
    df = pd.read_csv(DATA, parse_dates=["Date/Time"])
    df = df.sort_values("ohlcv_pos").reset_index(drop=True)

    px = pd.read_csv(OHLCV, parse_dates=["timestamp"])
    n_bars = len(px)
    bar_logret = np.log(px["close"] / px["close"].shift(1)).fillna(0.0).values

    # ---- label-overlap diagnostics ----
    t0_all = df["ohlcv_pos"].values.astype(int)
    t1_all = np.minimum(t0_all + df["bars_to_exit"].values.astype(int), n_bars - 1)
    conc = num_co_events(n_bars, t0_all, t1_all)
    spanned = conc[conc > 0]
    avg_u_all = get_avg_uniqueness(n_bars, t0_all, t1_all)
    print("=== LABEL OVERLAP DIAGNOSTICS (triple-barrier, folder 05) ===")
    print(f"  signals: {len(df)}   bars_to_exit: min={df['bars_to_exit'].min()} "
          f"max={df['bars_to_exit'].max()} mean={df['bars_to_exit'].mean():.2f}")
    print(f"  mean concurrency over spanned bars : {spanned.mean():.4f}")
    print(f"  mean average uniqueness            : {avg_u_all.mean():.4f}")
    print(f"  effective sample size              : {avg_u_all.sum():.0f} of {len(df)} "
          f"({100*avg_u_all.mean():.1f}%)")
    print(f"  uniqueness range                   : {avg_u_all.min():.4f} - {avg_u_all.max():.4f}")
    print(f"  rows with uniqueness < 1.0         : {(avg_u_all < 0.999).sum()} "
          f"({100*(avg_u_all < 0.999).mean():.1f}%)")

    # ---- chronological split: test = 2025-01-01 onward, touched once ----
    is_test = (df["Date/Time"] >= "2025-01-01").values
    df["vol_regime"] = recompute_vol_regime(df, ~is_test)
    pool = df[~is_test].reset_index(drop=True)
    test = df[is_test].reset_index(drop=True)
    pool_pos = np.where(~is_test)[0]
    print(f"\n  training pool: {len(pool)} rows   test: {len(test)} rows "
          f"({test['Date/Time'].min().date()} -> {test['Date/Time'].max().date()})")

    scheme_names = list(build_weights(df, pool_pos[:10], n_bars, bar_logret)[0].keys())

    # ---- walk-forward on the training pool, purged ----
    n = len(pool)
    min_train_n = int(n * MIN_TRAIN_FRAC)
    fold_size = (n - min_train_n) // N_WF_FOLDS
    results = {s: [] for s in scheme_names}

    print(f"\n=== PURGED WALK-FORWARD ({N_WF_FOLDS} expanding folds) ===")
    for fold in range(N_WF_FOLDS):
        train_end = min_train_n + fold * fold_size
        val_end = train_end + fold_size if fold < N_WF_FOLDS - 1 else n

        val_start_bar = int(pool.iloc[train_end]["ohlcv_pos"])
        # PURGE: drop training rows whose label window reaches into the val fold
        tr_idx = np.arange(train_end)
        tr_t1 = pool.iloc[tr_idx]["ohlcv_pos"].values + pool.iloc[tr_idx]["bars_to_exit"].values
        keep = tr_t1 < val_start_bar
        tr_idx = tr_idx[keep]
        n_purged = train_end - len(tr_idx)

        fold_tr = pool.iloc[tr_idx]
        fold_vl = pool.iloc[train_end:val_end]
        if len(fold_tr) < 200 or len(fold_vl) < 50:
            continue

        # internal early-stopping slice = last 15% of the fold's training rows
        cut = int(len(fold_tr) * 0.85)
        inner_tr, inner_vl = fold_tr.iloc[:cut], fold_tr.iloc[cut:]

        weights, _ = build_weights(df, pool_pos[tr_idx[:cut]], n_bars, bar_logret)

        line = f"  fold {fold+1}: train={len(fold_tr)} (purged {n_purged}) val={len(fold_vl)}"
        for scheme in scheme_names:
            p = np.zeros(len(fold_vl))
            for seed in SEEDS:
                p += fit_eval(inner_tr, inner_tr[TARGET], weights[scheme],
                              inner_vl, inner_vl[TARGET],
                              fold_vl, fold_vl[TARGET], seed)
            p /= len(SEEDS)
            auc = roc_auc_score(fold_vl[TARGET], p)
            results[scheme].append(auc)
            line += f"\n      {scheme:32s} AUC={auc:.4f}"
        print(line, flush=True)

    print("\n=== WALK-FORWARD SUMMARY ===")
    summary = []
    for scheme in scheme_names:
        a = results[scheme]
        if a:
            print(f"  {scheme:32s} AUC = {np.mean(a):.4f} +/- {np.std(a):.4f}")
            summary.append(dict(scheme=scheme, wf_auc_mean=np.mean(a), wf_auc_std=np.std(a)))

    # ---- final: fit on full pool, evaluate ONCE on the untouched test set ----
    print("\n=== FINAL TEST (2025-01 -> 2026-07, touched once) ===")
    cut = int(len(pool) * 0.85)
    final_tr, final_vl = pool.iloc[:cut], pool.iloc[cut:]
    weights, _ = build_weights(df, pool_pos[:cut], n_bars, bar_logret)

    for row in summary:
        scheme = row["scheme"]
        p = np.zeros(len(test))
        for seed in SEEDS:
            p += fit_eval(final_tr, final_tr[TARGET], weights[scheme],
                          final_vl, final_vl[TARGET], test, test[TARGET], seed)
        p /= len(SEEDS)
        row["test_auc"] = roc_auc_score(test[TARGET], p)
        print(f"  {scheme:32s} TEST AUC = {row['test_auc']:.4f}")

    res = pd.DataFrame(summary)
    res.to_csv("uniqueness_weighting_results.csv", index=False)

    base = res[res["scheme"].str.startswith("B.")].iloc[0]
    print("\n=== DID AFML WEIGHTING BEAT THE PRODUCTION SCHEME? ===")
    for _, r in res.iterrows():
        if r["scheme"].startswith("B."):
            continue
        d_wf = r["wf_auc_mean"] - base["wf_auc_mean"]
        d_te = r["test_auc"] - base["test_auc"]
        verdict = "improvement" if d_wf > base["wf_auc_std"] else "within noise"
        print(f"  {r['scheme']:32s} wf {d_wf:+.4f}  test {d_te:+.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
