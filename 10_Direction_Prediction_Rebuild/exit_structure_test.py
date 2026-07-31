"""
Decisive diagnostic: is the strategy unprofitable because of its ENTRY
trigger, or because of its EXIT structure?

The main search found every config bleeding money at profit factor ~0.86,
and the pinescript's exits are strongly asymmetric (3% fixed stop vs a 1%
trailing stop that arms after only 2% of gain -- i.e. it cuts winners short
and lets losers run to a full 3%). That asymmetry alone could explain the
losses without the entry being bad at all.

So: take the best entry configs found, and sweep a MUCH wider exit space --
including disabling the fixed stop, disabling trailing entirely, and adding
a fixed take-profit -- on both periods. If no exit structure makes the entry
profitable out of sample, the entry trigger itself has no edge, and no
amount of risk-management tuning will save it.

An entry with genuine edge should be profitable under SOME sane exit rule.
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from optimize_winrate import load_price, make_signals, ORIGINAL, SEARCH_END
from strategy_sim import simulate_exits, summarize

FIXED_SLS = [1.0, 2.0, 3.0, 5.0, 8.0, 1000.0]   # 1000 = disabled
TRAIL_SLS = [0.0, 1.0, 2.0, 3.0, 5.0]           # 0 = disabled
GAIN_THS  = [1.0, 2.0, 3.0]
TPS       = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]      # 0 = disabled

MIN_TRADES = 100


def run(px, mask, buy, sell, fsl, tsl, gth, tp):
    c = np.ascontiguousarray(px["close"].values[mask])
    h = np.ascontiguousarray(px["high"].values[mask])
    l = np.ascontiguousarray(px["low"].values[mask])
    pnls = simulate_exits(c, h, l,
                          np.ascontiguousarray(buy[mask]), np.ascontiguousarray(sell[mask]),
                          fsl, tsl, gth, tp)
    ts = px["timestamp"].values[mask]
    n_days = (pd.Timestamp(ts[-1]) - pd.Timestamp(ts[0])).days
    return summarize(pnls, n_days)


def main():
    px = load_price()
    is_search = (px["timestamp"] < SEARCH_END).values
    is_valid = ~is_search

    res = pd.read_csv("winrate_analysis_final.csv")
    sig_keys = ["rsi_len", "stoch_len", "adx_len", "adx_th", "os_th", "k_th", "d_th"]
    entries = pd.concat([
        res.nlargest(10, "search_win_rate")[sig_keys],
        res.nlargest(10, "search_profit_factor")[sig_keys],
        pd.DataFrame([{k: ORIGINAL[k] for k in sig_keys}]),
    ]).drop_duplicates().reset_index(drop=True)
    print(f"testing {len(entries)} entry configs x "
          f"{len(FIXED_SLS)*len(TRAIL_SLS)*len(GAIN_THS)*len(TPS)} exit structures")

    rows = []
    for ei, er in entries.iterrows():
        buy, sell = make_signals(px["close"], px["high"], px["low"],
                                 int(er["rsi_len"]), int(er["stoch_len"]), int(er["adx_len"]),
                                 er["adx_th"], er["os_th"], er["k_th"], er["d_th"])
        for fsl, tsl, gth, tp in itertools.product(FIXED_SLS, TRAIL_SLS, GAIN_THS, TPS):
            s = run(px, is_search, buy, sell, fsl, tsl, gth, tp)
            if s["n_trades"] < MIN_TRADES:
                continue
            v = run(px, is_valid, buy, sell, fsl, tsl, gth, tp)
            if v["n_trades"] < 20:
                continue
            rows.append(dict(entry_id=ei, **{k: er[k] for k in sig_keys},
                             fixed_sl=fsl, trail_sl=tsl, gain_th=gth, tp=tp,
                             search_win=s["win_rate"], search_pf=s["profit_factor"],
                             search_pnl=s["total_pnl"], search_n=s["n_trades"],
                             search_freq=s["trades_per_day"],
                             val_win=v["win_rate"], val_pf=v["profit_factor"],
                             val_pnl=v["total_pnl"], val_n=v["n_trades"],
                             val_freq=v["trades_per_day"]))
        print(f"  entry {ei+1}/{len(entries)} done, {len(rows)} rows", flush=True)

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_csv("exit_structure_results.csv", index=False)
    print(f"\n{len(df)} (entry x exit) combinations evaluated on BOTH periods")

    print(f"\nprofitable in search    : {(df['search_pf']>1).mean()*100:.2f}%")
    print(f"profitable in validation: {(df['val_pf']>1).mean()*100:.2f}%")
    both = ((df["search_pf"] > 1) & (df["val_pf"] > 1)).sum()
    print(f"profitable in BOTH      : {both} / {len(df)}")

    m = df[["search_pf", "val_pf"]].dropna()
    r, p = spearmanr(m["search_pf"], m["val_pf"])
    print(f"\nspearman(search PF, validation PF) = {r:+.4f} p={p:.3g}")

    print("\n=== best exit structures by SEARCH PF, with validation result ===")
    cols = ["rsi_len", "stoch_len", "adx_len", "adx_th", "os_th", "k_th", "d_th",
            "fixed_sl", "trail_sl", "gain_th", "tp",
            "search_win", "search_pf", "search_pnl", "search_freq",
            "val_win", "val_pf", "val_pnl", "val_freq"]
    print(df.nlargest(12, "search_pf")[cols].to_string(index=False))

    print("\n=== configs profitable in BOTH periods (if any) ===")
    bothdf = df[(df["search_pf"] > 1) & (df["val_pf"] > 1)]
    if len(bothdf):
        print(bothdf.nlargest(15, "val_pf")[cols].to_string(index=False))
    else:
        print("  NONE -- no entry/exit combination is profitable in both periods.")

    print("\n=== best WIN RATE (validation), any exit structure ===")
    print(df.nlargest(8, "val_win")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
