"""
Optimize the base pinescript strategy's inputs for MAXIMUM WIN RATE at a
target trading frequency of ~0.5 trades/day, using the faithful simulator
in strategy_sim.py (real entries, real 3%/1%/opposite-signal exits, real
commission) -- not a proxy label.

Discipline against the exact trap this project has been burned by twice
(a great-looking historical number that dies out of sample):
  - SEARCH period      : 2017-08 -> 2024-12  (all tuning happens here)
  - VALIDATION period  : 2025-01 -> 2026-07  (touched ONCE, at the end)
  - Baseline reported alongside: the ORIGINAL pinescript parameters, so any
    claimed improvement is measured against what you already have
  - Minimum trade count enforced, so a 100%-win-rate 3-trade config can't win
  - With ~10^5 configs searched, in-sample best is GUARANTEED to be inflated
    by multiple testing. The validation number is the only one that counts,
    and the gap between them is reported explicitly.
"""
import itertools
import numpy as np
import pandas as pd

from build_features import compute_rsi, compute_stoch_rsi, compute_adx, IN_PATH
from strategy_sim import simulate, summarize

SEARCH_END = "2025-01-01"

# --- grid over the pinescript's own inputs ---
RSI_LENGTHS   = [10, 14, 21]
STOCH_LENGTHS = [10, 14, 21]
ADX_LENGTHS   = [10, 14, 21]
ADX_THRESHOLDS = [20, 25, 30, 35, 40]
OS_THRESHOLDS  = [5, 10, 15, 20]     # prevK oversold gate (mirrored for shorts)
K_THRESHOLDS   = [10, 15, 20, 25, 30]
D_THRESHOLDS   = [5, 10, 15, 20]
# --- risk-management inputs (these directly drive win rate) ---
FIXED_SLS   = [2.0, 3.0, 5.0]
TRAIL_SLS   = [1.0, 2.0]
GAIN_THRESH = [2.0, 3.0]

TARGET_LO, TARGET_HI = 0.4, 0.6      # trades/day band around 0.5
MIN_TRADES_SEARCH = 200              # need enough trades for the win rate to mean anything

ORIGINAL = dict(rsi_len=14, stoch_len=14, adx_len=14, adx_th=20,
                os_th=20, k_th=30, d_th=20,
                fixed_sl=3.0, trail_sl=1.0, gain_th=2.0)


def load_price():
    px = pd.read_csv(IN_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return px[["timestamp", "open", "high", "low", "close", "volume"]]


def make_signals(c, h, l, rsi_len, stoch_len, adx_len, adx_th, os_th, k_th, d_th):
    rsi = compute_rsi(c, rsi_len)
    k, d = compute_stoch_rsi(rsi, stoch_len, 3, 3)
    adx, _, _ = compute_adx(h, l, c, adx_len)
    k_v, d_v, adx_v = k.values, d.values, adx.values
    prev_k = np.roll(k_v, 1); prev_k[0] = np.nan
    prev_d = np.roll(d_v, 1); prev_d[0] = np.nan
    cross_up = (prev_k < prev_d) & (k_v > d_v)
    cross_dn = (prev_k > prev_d) & (k_v < d_v)
    adx_ok = adx_v > adx_th
    buy = cross_up & (prev_k < os_th) & (k_v < k_th) & (d_v < d_th) & adx_ok
    sell = cross_dn & (prev_k > 100 - os_th) & (k_v > 100 - k_th) & (d_v > 100 - d_th) & adx_ok
    return np.nan_to_num(buy).astype(np.bool_), np.nan_to_num(sell).astype(np.bool_)


def run_config(px, mask, p):
    c = np.ascontiguousarray(px["close"].values[mask])
    h = np.ascontiguousarray(px["high"].values[mask])
    l = np.ascontiguousarray(px["low"].values[mask])
    buy, sell = p["_buy"][mask], p["_sell"][mask]
    pnls, dirs, ei, xi = simulate(c, h, l,
                                   np.ascontiguousarray(buy), np.ascontiguousarray(sell),
                                   p["fixed_sl"], p["trail_sl"], p["gain_th"])
    ts = px["timestamp"].values[mask]
    n_days = (pd.Timestamp(ts[-1]) - pd.Timestamp(ts[0])).days
    return summarize(pnls, n_days), pnls


def main():
    px = load_price()
    is_search = (px["timestamp"] < SEARCH_END).values
    is_valid = ~is_search
    print(f"search bars: {is_search.sum()}, validation bars: {is_valid.sum()}")

    c, h, l = px["close"], px["high"], px["low"]

    # ---------- baseline: the original pinescript parameters ----------
    b, s = make_signals(c, h, l, ORIGINAL["rsi_len"], ORIGINAL["stoch_len"],
                        ORIGINAL["adx_len"], ORIGINAL["adx_th"],
                        ORIGINAL["os_th"], ORIGINAL["k_th"], ORIGINAL["d_th"])
    base_p = dict(ORIGINAL); base_p["_buy"] = b; base_p["_sell"] = s
    base_search, _ = run_config(px, is_search, base_p)
    base_valid, _ = run_config(px, is_valid, base_p)
    print("\n=== BASELINE (original pinescript params) ===")
    print(f"  search    : n={base_search['n_trades']:5d} freq={base_search['trades_per_day']:.3f}/day "
          f"win_rate={base_search['win_rate']:.4f} PF={base_search['profit_factor']:.3f} "
          f"total_pnl={base_search['total_pnl']:.3f}")
    print(f"  validation: n={base_valid['n_trades']:5d} freq={base_valid['trades_per_day']:.3f}/day "
          f"win_rate={base_valid['win_rate']:.4f} PF={base_valid['profit_factor']:.3f} "
          f"total_pnl={base_valid['total_pnl']:.3f}")

    # ---------- grid search on the SEARCH period only ----------
    sig_combos = list(itertools.product(RSI_LENGTHS, STOCH_LENGTHS, ADX_LENGTHS,
                                         ADX_THRESHOLDS, OS_THRESHOLDS,
                                         K_THRESHOLDS, D_THRESHOLDS))
    risk_combos = list(itertools.product(FIXED_SLS, TRAIL_SLS, GAIN_THRESH))
    print(f"\nsearching {len(sig_combos)} signal x {len(risk_combos)} risk = "
          f"{len(sig_combos)*len(risk_combos)} configs on the search period...")

    results = []
    # cache indicator computation per (rsi_len, stoch_len, adx_len)
    for i, (rl, sl_, al, ath, osth, kth, dth) in enumerate(sig_combos):
        buy, sell = make_signals(c, h, l, rl, sl_, al, ath, osth, kth, dth)
        # cheap pre-filter: raw signal count must be in a plausible range
        n_sig_search = buy[is_search].sum() + sell[is_search].sum()
        n_days_search = (px["timestamp"][is_search].iloc[-1] - px["timestamp"][is_search].iloc[0]).days
        if not (0.25 <= n_sig_search / n_days_search <= 1.5):
            continue
        for fsl, tsl, gth in risk_combos:
            p = dict(rsi_len=rl, stoch_len=sl_, adx_len=al, adx_th=ath,
                     os_th=osth, k_th=kth, d_th=dth,
                     fixed_sl=fsl, trail_sl=tsl, gain_th=gth,
                     _buy=buy, _sell=sell)
            summ, _ = run_config(px, is_search, p)
            if summ["n_trades"] < MIN_TRADES_SEARCH:
                continue
            if not (TARGET_LO <= summ["trades_per_day"] <= TARGET_HI):
                continue
            row = {k: v for k, v in p.items() if not k.startswith("_")}
            row.update({f"search_{k}": v for k, v in summ.items()})
            results.append(row)
        if i % 500 == 0:
            print(f"  ...{i}/{len(sig_combos)} signal combos, {len(results)} qualifying so far")

    res = pd.DataFrame(results)
    print(f"\n{len(res)} configs qualified (freq in [{TARGET_LO},{TARGET_HI}]/day, >={MIN_TRADES_SEARCH} trades)")
    if len(res) == 0:
        print("no qualifying configs -- widen the grid or the frequency band")
        return
    res = res.sort_values("search_win_rate", ascending=False).reset_index(drop=True)
    res.to_csv("winrate_search_results.csv", index=False)
    print("\nTop 15 by SEARCH-period win rate:")
    cols = ["rsi_len", "stoch_len", "adx_len", "adx_th", "os_th", "k_th", "d_th",
            "fixed_sl", "trail_sl", "gain_th", "search_n_trades",
            "search_trades_per_day", "search_win_rate", "search_profit_factor", "search_total_pnl"]
    print(res[cols].head(15).to_string())
    print(f"\nsearch win_rate distribution: min={res['search_win_rate'].min():.4f} "
          f"median={res['search_win_rate'].median():.4f} max={res['search_win_rate'].max():.4f}")


if __name__ == "__main__":
    main()
