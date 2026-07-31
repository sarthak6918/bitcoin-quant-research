"""
Search the pinescript strategy's own tunable inputs (RSI/Stoch/ADX lengths,
ADX threshold, StochRSI oversold/overbought thresholds) to find the
parameterization whose resulting BUY/SELL entries are MAXIMALLY CORRELATED
with our engineered features -- operationalized as: entries should land on
bars where our already-trained, causal, walk-forward model (built in
build_features.py / train_walkforward.py) is most convinced (|prob - 0.5|
largest), subject to a ~2 signals/day frequency cap.

Two-period discipline to avoid the exact overfitting trap this project has
already been burned by twice:
  - SEARCH period : 2021-05-06 -> 2024-04-08 (oof_predictions_pool.csv) --
    every candidate parameterization is scored ONLY here.
  - VALIDATION period : 2025-01-01 -> 2026-07-22 (final_holdout_predictions.csv)
    -- touched exactly once, AFTER the winning parameterization is locked in,
    to check the correlation isn't a search-period fluke.

Objective (search period only): mean conviction = mean(|prob - 0.5| * 2)
over all fired (BUY+SELL) bars, subject to frequency in [1.5, 2.5]
signals/day. Reported alongside: direction-agreement rate (does the
strategy's own long/short call match the model's implied direction) and
realized AUC of the strategy's call vs actual outcome on fired bars.
"""
import pandas as pd
import numpy as np
from build_features import compute_rsi, compute_stoch_rsi, compute_adx, IN_PATH
from sklearn.metrics import roc_auc_score

RSI_LENGTHS = [10, 14, 21]
STOCH_LENGTHS = [10, 14, 21]
ADX_LENGTHS = [10, 14, 21]
K_LEN, D_LEN = 3, 3
ADX_THRESHOLDS = [15, 20, 25, 30]
OS_THRESHOLDS = [15, 20, 25, 30]     # prevK oversold/overbought mirror
K_THRESHOLDS = [25, 30, 35, 40]
D_THRESHOLDS = [15, 20, 25, 30]

TARGET_FREQ_LOW, TARGET_FREQ_HIGH = 1.5, 2.5  # signals/day

def load_conviction():
    oof = pd.read_csv("oof_predictions_pool.csv", parse_dates=["timestamp"])
    hold = pd.read_csv("final_holdout_predictions.csv", parse_dates=["timestamp"])
    oof["split"] = "search"
    hold["split"] = "validation"
    conv = pd.concat([oof, hold]).sort_values("timestamp").reset_index(drop=True)
    return conv

def main():
    price = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    price = price.sort_values("timestamp").reset_index(drop=True)
    c, h, l = price["close"], price["high"], price["low"]

    conv = load_conviction()
    price_ext = price.merge(conv[["timestamp", "pred", "actual_direction", "split"]],
                             on="timestamp", how="left")

    is_search = (price_ext["split"] == "search").values
    is_valid = (price_ext["split"] == "validation").values
    pred_arr = price_ext["pred"].values
    actual_arr = price_ext["actual_direction"].values

    n_days_search = (conv[conv.split == "search"]["timestamp"].max() -
                      conv[conv.split == "search"]["timestamp"].min()).days

    results = []
    for rsi_len in RSI_LENGTHS:
        rsi = compute_rsi(c, rsi_len)
        for stoch_len in STOCH_LENGTHS:
            k, d = compute_stoch_rsi(rsi, stoch_len, K_LEN, D_LEN)
            k_v, d_v = k.values, d.values
            prev_k = np.roll(k_v, 1); prev_k[0] = np.nan
            prev_d = np.roll(d_v, 1); prev_d[0] = np.nan
            cross_up = (prev_k < prev_d) & (k_v > d_v)
            cross_dn = (prev_k > prev_d) & (k_v < d_v)
            for adx_len in ADX_LENGTHS:
                adx, _, _ = compute_adx(h, l, c, adx_len)
                adx_v = adx.values

                for adx_th in ADX_THRESHOLDS:
                    adx_ok = adx_v > adx_th
                    for os_th in OS_THRESHOLDS:
                        for k_th in K_THRESHOLDS:
                            for d_th in D_THRESHOLDS:
                                buy = cross_up & (prev_k < os_th) & (k_v < k_th) & (d_v < d_th) & adx_ok
                                sell = cross_dn & (prev_k > 100 - os_th) & (k_v > 100 - k_th) & (d_v > 100 - d_th) & adx_ok

                                buy_search = buy & is_search
                                sell_search = sell & is_search
                                n_fired_search = buy_search.sum() + sell_search.sum()
                                if n_days_search <= 0:
                                    continue
                                freq = n_fired_search / n_days_search
                                if not (TARGET_FREQ_LOW <= freq <= TARGET_FREQ_HIGH):
                                    continue

                                fired_mask = (buy_search | sell_search) & ~np.isnan(pred_arr)
                                if fired_mask.sum() < 30:
                                    continue
                                pred_f = pred_arr[fired_mask]
                                actual_f = actual_arr[fired_mask]
                                strat_dir = buy_search[fired_mask].astype(int)  # 1=expects up

                                conviction = np.abs(pred_f - 0.5) * 2
                                mean_conviction = conviction.mean()
                                agree = (strat_dir == (pred_f > 0.5)).mean()

                                if len(np.unique(actual_f)) == 2:
                                    auc = roc_auc_score(actual_f, strat_dir)
                                else:
                                    auc = np.nan

                                results.append(dict(
                                    rsi_len=rsi_len, stoch_len=stoch_len, adx_len=adx_len,
                                    adx_th=adx_th, os_th=os_th, k_th=k_th, d_th=d_th,
                                    freq=freq, n_fired=n_fired_search,
                                    mean_conviction=mean_conviction, agree_rate=agree, auc=auc,
                                ))

    res = pd.DataFrame(results)
    res.to_csv("strategy_search_results.csv", index=False)
    print(f"Evaluated {len(res)} candidates within frequency band")
    res_sorted = res.sort_values("mean_conviction", ascending=False)
    print(res_sorted.head(15).to_string())
    return res_sorted

if __name__ == "__main__":
    main()
