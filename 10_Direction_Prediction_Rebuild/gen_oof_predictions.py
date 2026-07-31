"""
Generate causal (walk-forward, purged/embargoed) out-of-fold predicted
probabilities for horizon n=1 across the training-pool test region, so we
have a continuous "model conviction" series to correlate candidate
strategy-input parameterizations against. Combined with
final_holdout_predictions.csv (2025-01 -> 2026-07, already causal/OOS),
this gives a multi-year conviction series with zero leakage anywhere.
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from train_walkforward import get_feature_cols, SEEDS, FEAT_PATH, WARMUP, purge_train

N = 1

def main():
    df = pd.read_csv(FEAT_PATH, parse_dates=["timestamp"])
    df = df.iloc[WARMUP:].reset_index(drop=True)
    feature_cols = get_feature_cols(df)
    label_col = f"label_dir_{N}"

    valid = df.dropna(subset=[label_col] + feature_cols).reset_index(drop=True)
    pool = valid[valid["timestamp"] < "2025-01-01"].reset_index(drop=True)

    n_rows = len(pool)
    embargo = N * 2
    n_folds = 5
    fold_starts = np.linspace(int(n_rows * 0.5), int(n_rows * 0.9), n_folds + 1).astype(int)

    all_preds = []
    for i in range(n_folds):
        test_start, test_end = fold_starts[i], fold_starts[i + 1]
        train_df = purge_train(pool, test_start, N, embargo)
        test_df = pool.iloc[test_start:test_end]
        X_train, y_train = train_df[feature_cols], train_df[label_col]
        X_test = test_df[feature_cols]

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

        fold_out = test_df[["timestamp", label_col]].copy()
        fold_out["pred"] = preds
        fold_out = fold_out.rename(columns={label_col: "actual_direction"})
        all_preds.append(fold_out)
        print(f"fold {i}: {test_df['timestamp'].min()} -> {test_df['timestamp'].max()}, n={len(test_df)}")

    oof = pd.concat(all_preds).reset_index(drop=True)
    oof.to_csv("oof_predictions_pool.csv", index=False)
    print(f"\nSaved {len(oof)} oof rows, {oof['timestamp'].min()} -> {oof['timestamp'].max()}")

if __name__ == "__main__":
    main()
