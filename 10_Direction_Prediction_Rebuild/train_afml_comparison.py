"""
Does the lecture-derived feature set actually beat the baseline?

Same purged/embargoed walk-forward protocol as train_walkforward.py, same
model, same seeds -- only the feature set changes. Anything else held fixed
so the comparison is clean.

  A. baseline  : the original 63 causal features
  B. +AFML     : baseline + fractionally differentiated log price (5 orders,
                 raw and vol-normalized) + CUSUM structural-break features
  C. AFML only : the FFD/CUSUM features alone, to see what they carry by
                 themselves
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from train_walkforward import SEEDS, WARMUP, purge_train

FEAT_PATH = "features_labeled_afml.csv"
HORIZONS = [1, 5, 20]
AFML_PREFIXES = ("ffd_logp_", "cusum_")


def feature_sets(df):
    exclude = {"timestamp", "close"}
    exclude |= {c for c in df.columns if c.startswith("label_dir_") or c.startswith("fwdret_")}
    allf = [c for c in df.columns if c not in exclude]
    afml = [c for c in allf if c.startswith(AFML_PREFIXES)]
    base = [c for c in allf if not c.startswith(AFML_PREFIXES)]
    return {"A. baseline": base, "B. baseline+AFML": base + afml, "C. AFML only": afml}


def wf_eval(df, n, feature_cols, n_folds=5, embargo_mult=2):
    label_col = f"label_dir_{n}"
    valid = df.dropna(subset=[label_col] + feature_cols).reset_index(drop=True)
    n_rows = len(valid)
    embargo = n * embargo_mult
    fold_starts = np.linspace(int(n_rows * 0.5), int(n_rows * 0.9), n_folds + 1).astype(int)

    aucs = []
    for i in range(n_folds):
        test_start, test_end = fold_starts[i], fold_starts[i + 1]
        train_df = purge_train(valid, test_start, n, embargo)
        test_df = valid.iloc[test_start:test_end]
        if len(train_df) < 500 or len(test_df) < 50:
            continue
        X_tr, y_tr = train_df[feature_cols], train_df[label_col]
        X_te, y_te = test_df[feature_cols], test_df[label_col]

        halflife = 180 * 24
        age = (len(train_df) - 1) - np.arange(len(train_df))
        w = 0.5 ** (age / halflife)

        preds = np.zeros(len(X_te))
        for seed in SEEDS:
            m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.03,
                                   loss_function="Logloss", eval_metric="AUC",
                                   random_seed=seed, verbose=False,
                                   allow_writing_files=False)
            m.fit(X_tr, y_tr, sample_weight=w)
            preds += m.predict_proba(X_te)[:, 1]
        preds /= len(SEEDS)
        aucs.append(roc_auc_score(y_te, preds))
    return aucs


def main():
    df = pd.read_csv(FEAT_PATH, parse_dates=["timestamp"])
    df = df.iloc[WARMUP:].reset_index(drop=True)
    pool = df[df["timestamp"] < "2025-01-01"].reset_index(drop=True)
    sets = feature_sets(df)
    for k, v in sets.items():
        print(f"{k}: {len(v)} features")

    rows = []
    for n in HORIZONS:
        print(f"\n=== horizon n={n} ===", flush=True)
        for name, cols in sets.items():
            aucs = wf_eval(pool, n, cols)
            if not aucs:
                continue
            m, s = float(np.mean(aucs)), float(np.std(aucs))
            rows.append(dict(horizon=n, feature_set=name, auc_mean=m, auc_std=s,
                             n_features=len(cols)))
            print(f"  {name:18s} AUC = {m:.4f} +/- {s:.4f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv("afml_feature_comparison.csv", index=False)

    print("\n=== DID THE AFML FEATURES HELP? ===")
    for n in HORIZONS:
        sub = res[res["horizon"] == n].set_index("feature_set")
        if "A. baseline" in sub.index and "B. baseline+AFML" in sub.index:
            a = sub.loc["A. baseline", "auc_mean"]
            b = sub.loc["B. baseline+AFML", "auc_mean"]
            sd = sub.loc["A. baseline", "auc_std"]
            verdict = "improvement" if (b - a) > sd else "within fold-to-fold noise"
            print(f"  n={n:2d}: {a:.4f} -> {b:.4f}  (delta {b-a:+.4f}, "
                  f"baseline fold sd {sd:.4f})  -> {verdict}")


if __name__ == "__main__":
    main()
