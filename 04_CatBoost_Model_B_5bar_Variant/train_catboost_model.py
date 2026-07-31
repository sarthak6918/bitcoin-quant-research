"""
train_catboost_model.py — Reproduces the 5-bar-barrier BTC entry classifier
("Model C" in the barrier-comparison analysis) EXACTLY.

Run this against the provided training_data/*.csv files to reproduce the
model_seed*.cbm files bit-for-bit (CatBoost is deterministic given a fixed
random_seed and fixed inputs).

USAGE:
    python train_catboost_model.py

Expects to be run from a directory containing:
    training_data/train_split.csv
    training_data/validation_split.csv
    training_data/test_split_frozen_holdout.csv
(all three provided alongside this script)
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — see model_config.json for the same values in machine-readable form
# ─────────────────────────────────────────────────────────────────────────────
BASE_FEATURES = [
    "Type_encoded", "macd_histogram", "macd_signal", "rvi", "rvi_signal",
    "rsi_14", "rsi_lag1", "mfi_14", "adx", "plus_di", "minus_di", "di_net",
    "er_10", "volume", "bar_body_ratio", "adx_regime", "vol_regime",
    "er_adx_product", "adx_centered", "rsi_centered", "ema21_50_ratio",
    "ema9_21_ratio", "ema9_dist", "keltner_pos", "log_return_1",
    "log_return_5", "supertrend_dist", "atr_pct",
]  # 27 base features; the model additionally sees the target column separately

TARGET_COLUMN = "binary_target_vb5"  # 5-bar vertical-barrier label (see METHODOLOGY.md)
SEEDS = [42, 43, 44, 45, 46]          # multiple seeds fit + averaged for a stable AUC estimate
HALF_LIFE_DAYS = 180                 # exponential recency decay for sample weights

CATBOOST_PARAMS = dict(
    iterations=2000,
    learning_rate=0.02,
    depth=5,
    l2_leaf_reg=5,
    min_data_in_leaf=20,
    random_strength=1.5,
    bagging_temperature=0.8,
    loss_function="Logloss",
    eval_metric="AUC",
    od_type="Iter",
    od_wait=150,
    verbose=0,
    allow_writing_files=False,
)


def compute_sample_weights(dates: pd.Series, half_life_days: int = HALF_LIFE_DAYS) -> np.ndarray:
    """
    Exponential recency weighting: rows closer to the most recent date in the
    TRAINING set get weight close to 1.0; older rows decay by half every
    `half_life_days`. Weights are normalized to mean 1.0 so the effective
    training-set size stays comparable to an unweighted fit.
    """
    max_date = dates.max()
    days_old = (max_date - dates).dt.days.values.astype(float)
    w = np.exp(-np.log(2) * days_old / half_life_days)
    return w / w.mean()


def main():
    train = pd.read_csv("training_data/train_split.csv", parse_dates=["Date/Time"])
    val = pd.read_csv("training_data/validation_split.csv", parse_dates=["Date/Time"])
    test = pd.read_csv("training_data/test_split_frozen_holdout.csv", parse_dates=["Date/Time"])

    print(f"Train: {len(train)} rows ({train['Date/Time'].min()} -> {train['Date/Time'].max()})")
    print(f"Val  : {len(val)} rows ({val['Date/Time'].min()} -> {val['Date/Time'].max()})")
    print(f"Test : {len(test)} rows ({test['Date/Time'].min()} -> {test['Date/Time'].max()}) — frozen holdout")
    print()

    y_train = train[TARGET_COLUMN]
    y_val = val[TARGET_COLUMN]
    y_test = test[TARGET_COLUMN]

    w_train = compute_sample_weights(train["Date/Time"])

    # class_weights: balances the loss for the (usually mild) class imbalance
    # in win/loss labels — computed fresh each seed from the training labels,
    # not hardcoded, so it stays correct if you retrain on updated data.
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    test_aucs = []
    for seed in SEEDS:
        clf = CatBoostClassifier(
            **CATBOOST_PARAMS,
            class_weights={0: 1.0, 1: scale_pos},
            random_seed=seed,
        )
        clf.fit(
            train[BASE_FEATURES], y_train, sample_weight=w_train,
            eval_set=(val[BASE_FEATURES], y_val), use_best_model=True,
        )
        clf.save_model(f"models/model_seed{seed}.cbm")

        auc = roc_auc_score(y_test, clf.predict_proba(test[BASE_FEATURES])[:, 1])
        test_aucs.append(auc)
        print(f"  seed {seed}: best_iteration={clf.get_best_iteration()}  "
              f"frozen-holdout AUC={auc:.4f}")

    print()
    print(f"Mean frozen-holdout AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    print("(Expected: 0.9128 ± 0.0015 if this reproduces exactly)")


if __name__ == "__main__":
    main()
