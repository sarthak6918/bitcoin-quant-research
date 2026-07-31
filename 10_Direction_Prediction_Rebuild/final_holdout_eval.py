"""
Final, single-touch evaluation on the untouched 2025-01-01 -> 2026-07-22
holdout, using the horizon (n=1 bar) selected from walk-forward on the
training pool alone. This holdout is touched exactly once, here.
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from train_walkforward import get_feature_cols, SEEDS, FEAT_PATH, WARMUP

BEST_N = 1

def main():
    df = pd.read_csv(FEAT_PATH, parse_dates=["timestamp"])
    df = df.iloc[WARMUP:].reset_index(drop=True)
    feature_cols = get_feature_cols(df)
    label_col = f"label_dir_{BEST_N}"

    valid = df.dropna(subset=[label_col] + feature_cols).reset_index(drop=True)
    train_df = valid[valid["timestamp"] < "2025-01-01"].reset_index(drop=True)
    test_df = valid[valid["timestamp"] >= "2025-01-01"].reset_index(drop=True)
    print(f"train: {len(train_df)} rows, test: {len(test_df)} rows "
          f"({test_df['timestamp'].min()} -> {test_df['timestamp'].max()})")

    X_train, y_train = train_df[feature_cols], train_df[label_col]
    X_test, y_test = test_df[feature_cols], test_df[label_col]

    halflife_bars = 180 * 24
    age = (len(train_df) - 1) - np.arange(len(train_df))
    w = 0.5 ** (age / halflife_bars)

    preds = np.zeros(len(X_test))
    importances = np.zeros(len(feature_cols))
    for seed in SEEDS:
        model = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.03,
            loss_function="Logloss", eval_metric="AUC",
            random_seed=seed, verbose=False, allow_writing_files=False,
        )
        model.fit(X_train, y_train, sample_weight=w)
        preds += model.predict_proba(X_test)[:, 1]
        importances += model.get_feature_importance()
    preds /= len(SEEDS)
    importances /= len(SEEDS)

    auc = roc_auc_score(y_test, preds)
    print(f"\nFINAL HOLDOUT AUC (2025-01 -> 2026-07, n={BEST_N} bar): {auc:.4f}")
    print(f"Prediction distribution: min={preds.min():.3f} max={preds.max():.3f} "
          f"mean={preds.mean():.3f} std={preds.std():.3f}")

    # monthly breakdown
    test_df = test_df.copy()
    test_df["pred"] = preds
    test_df["month"] = test_df["timestamp"].dt.to_period("M")
    print("\nMonthly AUC:")
    for month, g in test_df.groupby("month"):
        if g[label_col].nunique() < 2 or len(g) < 30:
            continue
        m_auc = roc_auc_score(g[label_col], g["pred"])
        print(f"  {month}: n={len(g):4d} AUC={m_auc:.4f}")

    imp_sorted = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    print("\nTop 15 feature importances:")
    for f, imp in imp_sorted[:15]:
        print(f"  {f}: {imp:.2f}")

    out = test_df[["timestamp", label_col, "pred"]].rename(columns={label_col: "actual_direction"})
    out.to_csv("final_holdout_predictions.csv", index=False)

if __name__ == "__main__":
    main()
