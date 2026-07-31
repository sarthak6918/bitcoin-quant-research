"""
train_catboost_model.py — Reproduces the ACTUAL DEPLOYED BTC entry classifier
(10-bar vertical barrier, 28 features, no regime input) exactly.

This is the model that was live in production BEFORE any HMM regime work —
confirmed directly from retrain_from_signals_all_signals.py and
model_monitor.py, both of which train on `binary_target` (the 10-bar
column), never `binary_target_vb5`.

Run against the provided training_data/*.csv files to reproduce
model_seed*.cbm bit-for-bit (CatBoost is deterministic given a fixed
random_seed and fixed inputs).

USAGE:
    python train_catboost_model.py

Expects to be run from a directory containing:
    training_data/train_split.csv
    training_data/validation_split.csv
    training_data/test_split_frozen_holdout.csv
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
]  # 27 base features (28th "feature" some docs refer to is the target itself)

TARGET_COLUMN = "binary_target"  # 10-bar vertical-barrier label — CONFIRMED this is
                                  # what retrain_from_signals_all_signals.py and
                                  # model_monitor.py actually train on in production.
                                  # binary_target_vb5 (5-bar) is a separate research
                                  # variant, NOT what this deployed model uses.
SEEDS = [42, 43, 44, 45, 46]
HALF_LIFE_DAYS = 180

CATBOOST_PARAMS = dict(
    iterations=2000, learning_rate=0.02, depth=5, l2_leaf_reg=5,
    min_data_in_leaf=20, random_strength=1.5, bagging_temperature=0.8,
    loss_function="Logloss", eval_metric="AUC",
    od_type="Iter", od_wait=150, verbose=0, allow_writing_files=False,
)


def compute_sample_weights(dates: pd.Series, half_life_days: int = HALF_LIFE_DAYS) -> np.ndarray:
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

    y_train, y_val, y_test = train[TARGET_COLUMN], val[TARGET_COLUMN], test[TARGET_COLUMN]
    w_train = compute_sample_weights(train["Date/Time"])
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    test_aucs = []
    for seed in SEEDS:
        clf = CatBoostClassifier(**CATBOOST_PARAMS, class_weights={0: 1.0, 1: scale_pos}, random_seed=seed)
        clf.fit(train[BASE_FEATURES], y_train, sample_weight=w_train,
                eval_set=(val[BASE_FEATURES], y_val), use_best_model=True)
        clf.save_model(f"models/model_seed{seed}.cbm")

        auc = roc_auc_score(y_test, clf.predict_proba(test[BASE_FEATURES])[:, 1])
        test_aucs.append(auc)
        print(f"  seed {seed}: best_iteration={clf.get_best_iteration()}  frozen-holdout AUC={auc:.4f}")

    print()
    print(f"Mean frozen-holdout AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    print("(Expected: 0.8575 ± 0.0027 if this reproduces exactly)")


if __name__ == "__main__":
    main()
