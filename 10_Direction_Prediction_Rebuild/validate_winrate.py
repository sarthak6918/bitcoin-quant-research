"""
Take the top configs from optimize_winrate.py's SEARCH-period grid and
evaluate them ONCE on the untouched 2025-01 -> 2026-07 validation period.

With ~10^5 configs searched, the in-sample winner's win rate is inflated by
multiple testing essentially by construction. This script measures how much
of it survives out of sample, which is the only number worth acting on.

It also reports:
  - the ORIGINAL pinescript params on both periods (the thing to beat)
  - the search->validation DECAY for the top-N configs, so we can see whether
    the search found real structure or just fitted noise
  - a rank-correlation between search win rate and validation win rate across
    all qualifying configs. If that correlation is ~0, the search has no
    predictive power and NO config from it should be trusted.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from optimize_winrate import (load_price, make_signals, run_config, ORIGINAL,
                               SEARCH_END, TARGET_LO, TARGET_HI)

TOP_N = 25


SIG_KEYS = ["rsi_len", "stoch_len", "adx_len", "adx_th", "os_th", "k_th", "d_th"]


def evaluate_all(px, is_search, is_valid, res):
    """
    Evaluate every qualifying config out of sample, caching the (expensive)
    indicator/signal computation per unique signal-parameter tuple -- the
    risk parameters (SL/trailing/gain) don't change the signals, only the
    simulation, so recomputing indicators per row would be ~12x wasted work.
    """
    rows = []
    grouped = res.groupby(SIG_KEYS, sort=False)
    total = len(grouped)
    for gi, (key, grp) in enumerate(grouped):
        rl, sl_, al, ath, osth, kth, dth = key
        buy, sell = make_signals(px["close"], px["high"], px["low"],
                                 int(rl), int(sl_), int(al), ath, osth, kth, dth)
        for idx, row in grp.iterrows():
            p = dict(fixed_sl=row["fixed_sl"], trail_sl=row["trail_sl"],
                     gain_th=row["gain_th"], _buy=buy, _sell=sell)
            s_summ, _ = run_config(px, is_search, p)
            v_summ, _ = run_config(px, is_valid, p)
            rows.append(dict(
                idx=idx,
                search_win_rate=s_summ["win_rate"], val_win_rate=v_summ["win_rate"],
                search_pf=s_summ["profit_factor"], val_pf=v_summ["profit_factor"],
                search_pnl=s_summ["total_pnl"], val_pnl=v_summ["total_pnl"],
                val_n_trades=v_summ["n_trades"], val_freq=v_summ["trades_per_day"],
            ))
        if gi % 100 == 0:
            print(f"  ...{gi}/{total} unique signal combos", flush=True)
    return pd.DataFrame(rows).set_index("idx").sort_index()


def main():
    px = load_price()
    is_search = (px["timestamp"] < SEARCH_END).values
    is_valid = ~is_search

    res = pd.read_csv("winrate_search_results.csv")
    print(f"{len(res)} qualifying configs from the search")

    # ---- baseline ----
    b, s = make_signals(px["close"], px["high"], px["low"], ORIGINAL["rsi_len"],
                        ORIGINAL["stoch_len"], ORIGINAL["adx_len"], ORIGINAL["adx_th"],
                        ORIGINAL["os_th"], ORIGINAL["k_th"], ORIGINAL["d_th"])
    bp = dict(ORIGINAL); bp["_buy"] = b; bp["_sell"] = s
    base_s, _ = run_config(px, is_search, bp)
    base_v, _ = run_config(px, is_valid, bp)
    print("\n=== BASELINE (original pinescript params) ===")
    print(f"  search    : n={base_s['n_trades']:5d} freq={base_s['trades_per_day']:.3f} "
          f"win={base_s['win_rate']:.4f} PF={base_s['profit_factor']:.3f} pnl={base_s['total_pnl']:+.3f}")
    print(f"  validation: n={base_v['n_trades']:5d} freq={base_v['trades_per_day']:.3f} "
          f"win={base_v['win_rate']:.4f} PF={base_v['profit_factor']:.3f} pnl={base_v['total_pnl']:+.3f}")

    # ---- evaluate ALL qualifying configs out of sample (for the rank-corr check) ----
    print(f"\nevaluating all {len(res)} configs on the validation period...", flush=True)
    res = res.reset_index(drop=True)
    val = evaluate_all(px, is_search, is_valid, res)
    # drop columns the search csv already carries, else concat creates
    # duplicate names and downstream `full[col]` returns a 2-D frame
    val = val.drop(columns=[c for c in val.columns if c in res.columns])
    full = pd.concat([res, val], axis=1)
    full.to_csv("winrate_validation_results.csv", index=False)

    ok = full.dropna(subset=["search_win_rate", "val_win_rate"])
    ok = ok[ok["val_n_trades"] >= 20]
    rho, pval = spearmanr(ok["search_win_rate"], ok["val_win_rate"])
    print(f"\n=== DOES THE SEARCH GENERALIZE? ===")
    print(f"  Spearman rank corr (search win rate vs validation win rate) over "
          f"{len(ok)} configs: rho={rho:.4f}, p={pval:.4g}")
    if pval > 0.05 or rho <= 0:
        print("  -> NO significant positive relationship. The search-period ranking")
        print("     carries no information about out-of-sample win rate; picking the")
        print("     in-sample best is picking noise.")
    else:
        print("  -> Significant positive relationship; search ranking carries signal.")

    top = full.sort_values("search_win_rate", ascending=False).head(TOP_N)
    print(f"\n=== TOP {TOP_N} BY SEARCH WIN RATE, WITH THEIR VALIDATION RESULT ===")
    cols = ["rsi_len", "stoch_len", "adx_len", "adx_th", "os_th", "k_th", "d_th",
            "fixed_sl", "trail_sl", "gain_th",
            "search_win_rate", "val_win_rate", "val_n_trades", "val_freq",
            "search_pf", "val_pf", "val_pnl"]
    print(top[cols].to_string(index=False))
    print(f"\n  mean search win rate (top {TOP_N}): {top['search_win_rate'].mean():.4f}")
    print(f"  mean valid  win rate (top {TOP_N}): {top['val_win_rate'].mean():.4f}")
    print(f"  DECAY: {top['search_win_rate'].mean() - top['val_win_rate'].mean():+.4f}")
    print(f"\n  validation win rate across ALL qualifying configs: "
          f"mean={ok['val_win_rate'].mean():.4f} median={ok['val_win_rate'].median():.4f} "
          f"max={ok['val_win_rate'].max():.4f}")


if __name__ == "__main__":
    main()
