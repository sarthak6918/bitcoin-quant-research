"""
walkforward_5bar_regime.py — Walk-forward validated CatBoost model, 5-bar
vertical barrier, HMM regime features included, test set = the most recent
700 signals (never touched until the single final evaluation).

Lookahead-bias checklist (each verified, not assumed):
  1. Features: atr_pct/ema9_dist/keltner_pos/supertrend_dist rebuilt on the
     RAW signal-candle price (matching the live bot exactly), not the
     corrected_entry_price basis the original historical file used.
  2. Regime features: filtered_prob_state0-4 are the CAUSAL forward-algorithm
     probabilities (never the smoothed/Viterbi decode) -- confirmed at the
     point these were computed, reused unchanged here.
  3. Labels: 5-bar triple barrier, direction from the raw Signal column
     (not execution_signal), confirmed formula (max(2*ATR%,1.0) barriers).
  4. vol_regime bins: recomputed from the TRAINING POOL ONLY (not the full
     dataset, not the test set) -- avoids leaking future volatility
     quantile information into a "current" feature.
  5. Walk-forward folds: strictly expanding window, chronological, no
     shuffling -- each fold's validation data is entirely AFTER its
     training data.
  6. Final test set (last 700 signals) is never touched until the single
     final AUC computation -- no threshold or hyperparameter peeking.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

BASE_FEATURES = [
    "Type_encoded", "macd_histogram", "macd_signal", "rvi", "rvi_signal",
    "rsi_14", "rsi_lag1", "mfi_14", "adx", "plus_di", "minus_di", "di_net",
    "er_10", "volume", "bar_body_ratio", "adx_regime", "vol_regime",
    "er_adx_product", "adx_centered", "rsi_centered", "ema21_50_ratio",
    "ema9_21_ratio", "ema9_dist", "keltner_pos", "log_return_1",
    "log_return_5", "supertrend_dist", "atr_pct",
]
REGIME_FEATURES = [f"filtered_prob_state{i}" for i in range(5)]
ALL_FEATURES = BASE_FEATURES + REGIME_FEATURES  # 32 features total
TARGET = "binary_target_final"
HALF_LIFE_DAYS = 180
N_WF_FOLDS = 5
MIN_TRAIN_FRAC = 0.40
FINAL_SEEDS = [42, 43, 44, 45, 46]


def compute_sample_weights(dates, half_life_days=HALF_LIFE_DAYS):
    max_date = dates.max()
    days_old = (max_date - dates).dt.days.values.astype(float)
    w = np.exp(-np.log(2) * days_old / half_life_days)
    return w / w.mean()


def recompute_vol_regime(df, fit_mask, atr_col="atr_pct"):
    """Recompute vol_regime bins from ONLY the rows in fit_mask (no leakage)."""
    bins = np.quantile(df.loc[fit_mask, atr_col].values, [0.25, 0.50, 0.75])
    return np.digitize(df[atr_col].values, bins=bins), bins


def fit_one(X_tr, y_tr, w_tr, X_vl, y_vl, seed):
    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    clf = CatBoostClassifier(
        iterations=2000, learning_rate=0.02, depth=5, l2_leaf_reg=5,
        min_data_in_leaf=20, random_strength=1.5, bagging_temperature=0.8,
        loss_function="Logloss", eval_metric="AUC",
        class_weights={0: 1.0, 1: scale_pos},
        random_seed=seed, od_type="Iter", od_wait=150,
        verbose=0, allow_writing_files=False,
    )
    clf.fit(X_tr[ALL_FEATURES], y_tr, sample_weight=w_tr,
            eval_set=(X_vl[ALL_FEATURES], y_vl), use_best_model=True)
    return clf


def main():
    train_pool = pd.read_csv("/tmp/wf_train_pool.csv", parse_dates=["Date/Time"])
    test = pd.read_csv("/tmp/wf_test_700.csv", parse_dates=["Date/Time"])

    print(f"Training pool: {len(train_pool)} rows ({train_pool['Date/Time'].min()} -> {train_pool['Date/Time'].max()})")
    print(f"Test (last 700, untouched until final eval): {len(test)} rows "
          f"({test['Date/Time'].min()} -> {test['Date/Time'].max()})")
    print(f"Features: {len(ALL_FEATURES)} ({len(BASE_FEATURES)} base + {len(REGIME_FEATURES)} regime)")
    print("=" * 78)

    # ---- Recompute vol_regime bins from the training pool ONLY ----
    full = pd.concat([train_pool, test], ignore_index=True)
    fit_mask = np.arange(len(full)) < len(train_pool)
    full["vol_regime"], vol_bins = recompute_vol_regime(full, fit_mask)
    print(f"\nvol_regime bins recomputed from training-pool-only atr_pct quantiles: {vol_bins.round(4)}")
    train_pool = full.iloc[:len(train_pool)].reset_index(drop=True)
    test = full.iloc[len(train_pool):].reset_index(drop=True)

    # ---- Walk-forward validation on the training pool (expanding window) ----
    print("\n" + "=" * 78)
    print(f"WALK-FORWARD VALIDATION ({N_WF_FOLDS} expanding-window folds on training pool)")
    print("=" * 78)
    n = len(train_pool)
    min_train_n = int(n * MIN_TRAIN_FRAC)
    remaining = n - min_train_n
    fold_size = remaining // N_WF_FOLDS

    wf_aucs = []
    for fold in range(N_WF_FOLDS):
        train_end = min_train_n + fold * fold_size
        val_end = train_end + fold_size if fold < N_WF_FOLDS - 1 else n
        fold_train = train_pool.iloc[:train_end]
        fold_val = train_pool.iloc[train_end:val_end]

        w_tr = compute_sample_weights(fold_train["Date/Time"])
        clf = fit_one(fold_train, fold_train[TARGET], w_tr, fold_val, fold_val[TARGET], seed=42)
        p_val = clf.predict_proba(fold_val[ALL_FEATURES])[:, 1]
        auc = roc_auc_score(fold_val[TARGET], p_val)
        wf_aucs.append(auc)
        print(f"  Fold {fold+1}: train={len(fold_train):4d} rows (ends {fold_train['Date/Time'].max().date()})  "
              f"val={len(fold_val):4d} rows ({fold_val['Date/Time'].min().date()} -> {fold_val['Date/Time'].max().date()})  "
              f"AUC={auc:.4f}")

    print(f"\nWalk-forward mean AUC: {np.mean(wf_aucs):.4f} ± {np.std(wf_aucs):.4f}")

    # ---- Final model: fit on the FULL training pool, evaluate ONCE on the untouched test set ----
    print("\n" + "=" * 78)
    print("FINAL MODEL — fit on full training pool, evaluated once on last-700 test set")
    print("=" * 78)
    # internal chronological validation slice for early stopping (last 15% of training pool)
    vs = int(len(train_pool) * 0.85)
    final_train = train_pool.iloc[:vs]
    final_val = train_pool.iloc[vs:]
    w_final_train = compute_sample_weights(final_train["Date/Time"])

    test_aucs = []
    all_probs = []
    for seed in FINAL_SEEDS:
        clf = fit_one(final_train, final_train[TARGET], w_final_train,
                      final_val, final_val[TARGET], seed)
        p_test = clf.predict_proba(test[ALL_FEATURES])[:, 1]
        all_probs.append(p_test)
        auc = roc_auc_score(test[TARGET], p_test)
        test_aucs.append(auc)
        print(f"  seed {seed}: best_iteration={clf.get_best_iteration()}  FINAL TEST AUC={auc:.4f}")

    print(f"\nMEAN FINAL TEST AUC (last 700 signals, 5-bar barrier, +HMM regime): "
          f"{np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")

    # breakdown by source within the test set
    test = test.copy()
    test["pred_prob"] = np.mean(all_probs, axis=0)
    print("\nTest set AUC by source:")
    for src, g in test.groupby("source"):
        if g[TARGET].nunique() < 2:
            print(f"  {src}: n={len(g)}  single-class")
            continue
        print(f"  {src}: n={len(g)}  AUC={roc_auc_score(g[TARGET], g['pred_prob']):.4f}")

    test.to_csv("/tmp/wf_test_700_with_preds.csv", index=False)


if __name__ == "__main__":
    main()
