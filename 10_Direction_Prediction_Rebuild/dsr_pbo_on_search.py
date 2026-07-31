"""
Lecture 6 applied to this project's own strategy search.

The grid search in optimize_winrate.py tried 129,600 configurations and the
best one showed profit factor 1.097 on the search period. Lecture 6's whole
point is that such a number is meaningless without correcting for how many
trials produced it: with enough trials, a spectacular best-Sharpe is
GUARANTEED even when no configuration has any real edge.

Two corrections applied here:
  1. Deflated Sharpe Ratio  -- is the best config's Sharpe significant once
     deflated for 129,600 trials, non-normal returns, and sample length?
  2. Probability of Backtest Overfitting (CSCV) -- if we select the best
     config in-sample, how often does it land in the bottom half OOS?
"""
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from optimize_winrate import load_price, make_signals, SEARCH_END
from strategy_sim import simulate
from afml_lectures import (
    deflated_sharpe_ratio, expected_max_sharpe, prob_backtest_overfitting,
)

N_TRIALS_ACTUAL = 129600   # full grid searched in optimize_winrate.py
N_STRATS_FOR_PBO = 120     # sample of configs to build the PBO matrix


def per_trade_returns(px, mask, row):
    buy, sell = make_signals(px["close"], px["high"], px["low"],
                             int(row["rsi_len"]), int(row["stoch_len"]),
                             int(row["adx_len"]), row["adx_th"], row["os_th"],
                             row["k_th"], row["d_th"])
    c = np.ascontiguousarray(px["close"].values[mask])
    h = np.ascontiguousarray(px["high"].values[mask])
    l = np.ascontiguousarray(px["low"].values[mask])
    buy = np.asarray(buy)
    sell = np.asarray(sell)
    pnls, _dirs, _ei, _xi = simulate(c, h, l,
                                     np.ascontiguousarray(buy[mask]),
                                     np.ascontiguousarray(sell[mask]),
                                     row["fixed_sl"], row["trail_sl"], row["gain_th"])
    return pnls


def main():
    px = load_price()
    is_search = (px["timestamp"] < SEARCH_END).values

    res = pd.read_csv("winrate_analysis_final.csv")
    best = res.nlargest(1, "search_profit_factor").iloc[0]

    # ---------------- sample of trials, used for BOTH the cross-trial
    # Sharpe variance (needed to scale SR0 correctly) and the PBO matrix ----
    sample = res.sample(n=min(N_STRATS_FOR_PBO, len(res)), random_state=42)
    series, sample_sharpes = [], []
    min_len = None
    for _, row in sample.iterrows():
        rr = per_trade_returns(px, is_search, row)
        if len(rr) < 200:
            continue
        sd = np.std(rr)
        if sd > 0:
            sample_sharpes.append(np.mean(rr) / sd)
        series.append(rr)
        min_len = len(rr) if min_len is None else min(min_len, len(rr))
    if not series:
        print("not enough trades to analyse")
        return

    # DSR's SR0 must be scaled by the VARIANCE OF THE SHARPE ESTIMATES ACROSS
    # TRIALS (Bailey & Lopez de Prado). Using the default 1.0 would compare a
    # per-trade Sharpe against a benchmark on a completely different scale.
    var_sharpe = float(np.var(sample_sharpes, ddof=1))
    print(f"cross-trial Sharpe dispersion from {len(sample_sharpes)} sampled configs: "
          f"var={var_sharpe:.6f}  sd={np.sqrt(var_sharpe):.4f}")

    # ---------------- 1. Deflated Sharpe Ratio on the best config ----------
    r = per_trade_returns(px, is_search, best)
    sr = np.mean(r) / np.std(r)
    sk = float(skew(r))
    ku = float(kurtosis(r, fisher=False))

    print("\n=== DEFLATED SHARPE RATIO (Lecture 6) ===")
    print(f"Best config by search profit factor (PF={best['search_profit_factor']:.4f}):")
    print(f"  trades={len(r)}  per-trade Sharpe={sr:.4f}  skew={sk:.3f}  kurtosis={ku:.3f}")

    for n_trials in (1, 100, 10000, N_TRIALS_ACTUAL):
        dsr, sr0 = deflated_sharpe_ratio(sr, n_trials, len(r), sk, ku, var_sharpe)
        flag = "SIGNIFICANT" if dsr > 0.95 else "not significant"
        print(f"  n_trials={n_trials:7d}: benchmark SR0={sr0:.4f}  DSR={dsr:.4f}  -> {flag}")

    sr0_actual = expected_max_sharpe(N_TRIALS_ACTUAL, var_sharpe)
    print(f"\n  Under the null of ZERO skill, searching {N_TRIALS_ACTUAL:,} configs is")
    print(f"  expected to produce a best per-trade Sharpe of {sr0_actual:.4f} by luck alone.")
    print(f"  We observed {sr:.4f}.  Observed/expected-by-luck = {sr/sr0_actual:.3f}x")

    # ---------------- 2. PBO via CSCV ----------------
    print(f"\n=== PROBABILITY OF BACKTEST OVERFITTING (CSCV, Lecture 6) ===")
    M = np.column_stack([s[:min_len] for s in series])
    print(f"  matrix: {M.shape[0]} trades x {M.shape[1]} strategies")

    pbo = prob_backtest_overfitting(M, n_splits=12)
    print(f"  PBO = {pbo:.4f}")
    if pbo > 0.5:
        print("  -> Selecting the in-sample best is WORSE than random: it lands in the")
        print("     bottom half out of sample more often than not.")
    elif pbo > 0.3:
        print("  -> Selecting the in-sample best is close to a coin flip out of sample.")
    else:
        print("  -> In-sample selection carries real out-of-sample information.")

    pd.DataFrame([dict(observed_sharpe=sr, skew=sk, kurtosis=ku, n_trades=len(r),
                       sr0_at_actual_trials=sr0_actual,
                       dsr_at_actual_trials=deflated_sharpe_ratio(
                           sr, N_TRIALS_ACTUAL, len(r), sk, ku)[0],
                       pbo=pbo)]).to_csv("dsr_pbo_results.csv", index=False)
    print("\nsaved dsr_pbo_results.csv")


if __name__ == "__main__":
    main()
