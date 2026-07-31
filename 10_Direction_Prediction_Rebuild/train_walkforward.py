"""
Purged, embargoed walk-forward CV + CatBoost training for multi-horizon
direction prediction, on the full causal feature set from build_features.py.

Design (mirrors the discipline already validated elsewhere in this project):
  - Chronological splits only, never shuffled
  - Final holdout = everything from 2025-01-01 onward, touched exactly once,
    only after the horizon/hyperparameter choice is locked in from walk-forward
  - Training pool = everything before 2025-01-01
  - Walk-forward: 5 expanding-window folds inside the training pool
  - PURGE/EMBARGO: because label_dir_n uses close[t+n], any training row whose
    label window [t, t+n] overlaps the test fold's start is dropped from
    training for that fold (purging), and an additional n-bar gap is left
    between train and test (embargo) to kill residual autocorrelation leakage
  - 5 seeds per fold, predictions averaged
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

FEAT_PATH = "features_labeled.csv"
WARMUP = 200
SEEDS = [42, 43, 44, 45, 46]
HORIZONS = [1, 2, 3, 5, 10, 20, 50]

FEATURE_COLS = None  # filled in main()

def get_feature_cols(df):
    exclude = {"timestamp", "close"}
    exclude |= {c for c in df.columns if c.startswith("label_dir_") or c.startswith("fwdret_")}
    return [c for c in df.columns if c not in exclude]

def purge_train(train_df, test_start_idx, n, embargo):
    # drop training rows whose label window [t, t+n] runs into [test_start_idx - embargo, ...)
    cutoff = test_start_idx - embargo
    return train_df[train_df.index < cutoff]

def walk_forward_eval(df, n, feature_cols, n_folds=5, embargo_mult=2):
    label_col = f"label_dir_{n}"
    valid = df.dropna(subset=[label_col] + feature_cols).reset_index(drop=True)
    n_rows = len(valid)
    embargo = n * embargo_mult

    fold_starts = np.linspace(int(n_rows * 0.5), int(n_rows * 0.9), n_folds + 1).astype(int)
    fold_aucs = []
    for i in range(n_folds):
        test_start = fold_starts[i]
        test_end = fold_starts[i + 1]
        train_df = purge_train(valid, test_start, n, embargo)
        test_df = valid.iloc[test_start:test_end]
        if len(train_df) < 500 or len(test_df) < 50:
            continue

        X_train, y_train = train_df[feature_cols], train_df[label_col]
        X_test, y_test = test_df[feature_cols], test_df[label_col]

        # recency weighting: 180-day half-life in bars (hourly -> *24)
        halflife_bars = 180 * 24
        age = (len(train_df) - 1) - np.arange(len(train_df))
        w = 0.5 ** (age / halflife_bars)

        preds = np.zeros(len(X_test))
        for seed in SEEDS:
            model = CatBoostClassifier(
                iterations=400, depth=6, learning_rate=0.03,
                loss_function="Logloss", eval_metric="AUC",
                random_seed=seed, verbose=False, allow_writing_files=False,
            )
            model.fit(X_train, y_train, sample_weight=w)
            preds += model.predict_proba(X_test)[:, 1]
        preds /= len(SEEDS)
        auc = roc_auc_score(y_test, preds)
        fold_aucs.append(auc)
        print(f"  horizon={n} fold={i} train_n={len(train_df)} test_n={len(test_df)} AUC={auc:.4f}")
    return fold_aucs

def main():
    df = pd.read_csv(FEAT_PATH, parse_dates=["timestamp"])
    df = df.iloc[WARMUP:].reset_index(drop=True)

    training_pool = df[df["timestamp"] < "2025-01-01"].reset_index(drop=True)
    print(f"Training pool: {len(training_pool)} rows ({training_pool['timestamp'].min()} -> {training_pool['timestamp'].max()})")

    feature_cols = get_feature_cols(df)
    print(f"{len(feature_cols)} feature columns")

    results = {}
    for n in HORIZONS:
        print(f"\n=== Horizon {n} bars ===")
        aucs = walk_forward_eval(training_pool, n, feature_cols)
        if aucs:
            results[n] = (np.mean(aucs), np.std(aucs))
            print(f"  MEAN AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    print("\n=== SUMMARY (walk-forward, training pool only, purged/embargoed) ===")
    for n, (m, s) in results.items():
        print(f"  n={n:3d} bars: AUC = {m:.4f} +/- {s:.4f}")

if __name__ == "__main__":
    main()
