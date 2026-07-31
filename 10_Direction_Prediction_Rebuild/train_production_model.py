"""
Train the final production ensemble (n=1 bar horizon, the best-performing
and most stable config from walk-forward) on ALL available history through
the last row of features_labeled.csv, and persist it for live_inference.py.

Regime (HMM) probability columns are dropped -- confirmed to cost 0.0 AUC
(0.5397 without vs 0.5382 with, on the 2025-2026 holdout) and would require
a live HMM decoder dependency (hmmlearn) that isn't installable in this
environment. See feature_lib.py docstring.

This is a PRODUCTION artifact trained on the full dataset, not a walk-forward
research fold -- expect its in-sample behavior to look better than the
walk-forward numbers; what matters is that walk-forward already gave the
honest expectation (~0.54-0.57 AUC) for how this generalizes.
"""
import json
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from train_walkforward import get_feature_cols, SEEDS, FEAT_PATH, WARMUP

N = 1
OUT_DIR = "production_models"

def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(FEAT_PATH, parse_dates=["timestamp"])
    df = df.iloc[WARMUP:].reset_index(drop=True)
    feature_cols = [c for c in get_feature_cols(df) if not c.startswith("filtered_prob_state")]
    label_col = f"label_dir_{N}"

    valid = df.dropna(subset=[label_col] + feature_cols).reset_index(drop=True)
    print(f"Training on {len(valid)} rows, {valid['timestamp'].min()} -> {valid['timestamp'].max()}")

    X, y = valid[feature_cols], valid[label_col]
    halflife_bars = 180 * 24
    age = (len(valid) - 1) - np.arange(len(valid))
    w = 0.5 ** (age / halflife_bars)

    for seed in SEEDS:
        model = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.03,
            loss_function="Logloss", eval_metric="AUC",
            random_seed=seed, verbose=False, allow_writing_files=False,
        )
        model.fit(X, y, sample_weight=w)
        model.save_model(f"{OUT_DIR}/model_seed{seed}.cbm")
        print(f"  saved model_seed{seed}.cbm")

    # thresholds calibrated to ~2 signals/day from the combined causal
    # conviction series (oof pool 2021-2024 + true holdout 2025-2026)
    oof = pd.read_csv("oof_predictions_pool.csv", parse_dates=["timestamp"])
    hold = pd.read_csv("final_holdout_predictions.csv", parse_dates=["timestamp"])
    conv = pd.concat([oof["pred"], hold["pred"]])
    tail_frac = (2.0 / 24.0) / 2.0  # 2 signals/day combined, split symmetric
    lo_thresh = float(conv.quantile(tail_frac))
    hi_thresh = float(conv.quantile(1 - tail_frac))

    config = {
        "horizon_bars": N,
        "feature_cols": feature_cols,
        "seeds": SEEDS,
        "buy_threshold": hi_thresh,
        "sell_threshold": lo_thresh,
        "calibration_note": f"thresholds set from empirical quantiles (tail={tail_frac:.4f} each side) "
                             f"of causal OOS predictions 2021-2024 + 2025-2026, targeting ~2 signals/day combined",
        "min_warmup_hours": 220,
    }
    with open(f"{OUT_DIR}/production_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nbuy_threshold (prob >=): {hi_thresh:.4f}")
    print(f"sell_threshold (prob <=): {lo_thresh:.4f}")
    print(f"saved {OUT_DIR}/production_config.json")

if __name__ == "__main__":
    main()
