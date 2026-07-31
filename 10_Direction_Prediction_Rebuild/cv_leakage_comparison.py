"""
Quantify EXACTLY how much AUC the shuffled-CV leakage manufactures on this
project's real data, by scoring the identical model under four CV schemes:

  1. StratifiedKFold(shuffle=True)  -- the leaky standard practice AFML warns
                                       about (Ch.7 sec 7.3)
  2. KFold(shuffle=False)           -- contiguous folds, but no purge/embargo
  3. PurgedKFold(pct_embargo=0)     -- purging only (AFML sec 7.4.1)
  4. PurgedKFold(pct_embargo=0.01)  -- purging + embargo (AFML sec 7.4.2/7.4.3)

Run across horizons so the effect can be seen growing with label overlap:
at n=1 labels barely overlap, at n=50 each label shares 49 of its 50 bars
with its neighbour, so leakage should scale with n.

Also reports the EFFECTIVE sample size implied by AFML Ch.4 average
uniqueness -- i.e. how many genuinely independent observations the training
set actually contains, versus how many rows it appears to contain.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold

from afml import PurgedKFold, build_t1, get_avg_uniqueness, num_co_events
from train_walkforward import get_feature_cols, FEAT_PATH, WARMUP

HORIZONS = [1, 5, 20, 50]
N_SPLITS = 5
SEED = 42
SUBSAMPLE = 30000   # keep runtime sane; chronologically contiguous tail


def fit_score(X_tr, y_tr, X_te, y_te):
    m = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.03,
        loss_function="Logloss", random_seed=SEED,
        verbose=False, allow_writing_files=False,
    )
    m.fit(X_tr, y_tr)
    return roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])


def run_scheme(name, splitter, X, y):
    aucs = []
    for tr, te in splitter:
        if len(tr) < 500 or len(te) < 100:
            continue
        if len(np.unique(y.iloc[te])) < 2:
            continue
        aucs.append(fit_score(X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]))
    return np.mean(aucs), np.std(aucs), len(aucs)


def main():
    df = pd.read_csv(FEAT_PATH, parse_dates=["timestamp"])
    df = df.iloc[WARMUP:].reset_index(drop=True)
    feature_cols = get_feature_cols(df)

    rows = []
    for n in HORIZONS:
        label_col = f"label_dir_{n}"
        d = df.dropna(subset=[label_col] + feature_cols).reset_index(drop=True)
        d = d.iloc[-SUBSAMPLE:].reset_index(drop=True)

        X = d[feature_cols]
        y = d[label_col].astype(int)
        n_obs = len(d)

        # ---- AFML Ch.4: how many INDEPENDENT observations do we really have? ----
        start_idx = np.arange(n_obs)
        end_idx = np.minimum(start_idx + n, n_obs - 1)
        avg_u = get_avg_uniqueness(n_obs, start_idx, end_idx)
        conc = num_co_events(n_obs, start_idx, end_idx)
        eff_n = avg_u.sum()

        print(f"\n=== horizon n={n} bars  ({n_obs} rows) ===")
        print(f"  mean concurrency        : {conc.mean():.2f} overlapping labels per bar")
        print(f"  mean average uniqueness : {avg_u.mean():.4f}")
        print(f"  EFFECTIVE sample size   : {eff_n:.0f}  "
              f"({eff_n/n_obs*100:.1f}% of the {n_obs} rows)")

        t1 = build_t1(d["timestamp"], n)
        Xi = X.set_index(t1.index)

        schemes = {
            "1. StratifiedKFold(shuffle=True)  [LEAKY]":
                list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                     random_state=SEED).split(X, y)),
            "2. KFold(shuffle=False)":
                list(KFold(n_splits=N_SPLITS, shuffle=False).split(X)),
            "3. PurgedKFold (purge only)":
                list(PurgedKFold(n_splits=N_SPLITS, t1=t1, pct_embargo=0.0).split(Xi)),
            "4. PurgedKFold + 1% embargo":
                list(PurgedKFold(n_splits=N_SPLITS, t1=t1, pct_embargo=0.01).split(Xi)),
        }

        for name, splits in schemes.items():
            mean_auc, std_auc, k = run_scheme(name, splits, X, y)
            mean_train = int(np.mean([len(tr) for tr, _ in splits]))
            print(f"  {name:42s} AUC={mean_auc:.4f} +/- {std_auc:.4f} "
                  f"(train_n~{mean_train})")
            rows.append(dict(horizon=n, scheme=name, auc=mean_auc, std=std_auc,
                             folds=k, mean_train_n=mean_train,
                             mean_concurrency=conc.mean(),
                             mean_uniqueness=avg_u.mean(), effective_n=eff_n))

    res = pd.DataFrame(rows)
    res.to_csv("cv_leakage_comparison.csv", index=False)

    print("\n\n=== LEAKAGE MANUFACTURED BY SHUFFLING (AUC inflation) ===")
    piv = res.pivot(index="horizon", columns="scheme", values="auc")
    leaky = piv["1. StratifiedKFold(shuffle=True)  [LEAKY]"]
    honest = piv["4. PurgedKFold + 1% embargo"]
    for n in piv.index:
        print(f"  n={n:3d}: shuffled={leaky[n]:.4f}  purged+embargo={honest[n]:.4f}  "
              f"INFLATION = {leaky[n]-honest[n]:+.4f}")
    print(f"\nsaved cv_leakage_comparison.csv")


if __name__ == "__main__":
    main()
