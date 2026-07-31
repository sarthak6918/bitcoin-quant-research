"""
Analyze the saved search->validation results and answer the only question
that matters: does ANY of the search-period optimization survive out of
sample, or is it multiple-testing noise?

Reads winrate_validation_results.csv (produced by validate_winrate.py).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest

from optimize_winrate import load_price, make_signals, run_config, ORIGINAL, SEARCH_END


def clean(df):
    # read_csv dedupes duplicate headers with a .1 suffix; drop those
    return df[[c for c in df.columns if not c.endswith(".1")]]


def rho(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan, np.nan
    r, p = spearmanr(np.asarray(a)[m], np.asarray(b)[m])
    return float(r), float(p)


def main():
    full = clean(pd.read_csv("winrate_validation_results.csv"))
    px = load_price()
    is_search = (px["timestamp"] < SEARCH_END).values
    is_valid = ~is_search

    # ---------------- baseline ----------------
    b, s = make_signals(px["close"], px["high"], px["low"], ORIGINAL["rsi_len"],
                        ORIGINAL["stoch_len"], ORIGINAL["adx_len"], ORIGINAL["adx_th"],
                        ORIGINAL["os_th"], ORIGINAL["k_th"], ORIGINAL["d_th"])
    bp = dict(ORIGINAL); bp["_buy"] = b; bp["_sell"] = s
    base_s, _ = run_config(px, is_search, bp)
    base_v, _ = run_config(px, is_valid, bp)
    print("=== BASELINE (original pinescript params) ===")
    for nm, r in (("search", base_s), ("validation", base_v)):
        print(f"  {nm:10s}: n={r['n_trades']:5d} freq={r['trades_per_day']:.3f}/day "
              f"win={r['win_rate']:.4f} PF={r['profit_factor']:.3f} pnl={r['total_pnl']:+.3f}")

    ok = full[full["val_n_trades"] >= 20].copy()
    print(f"\n{len(full)} qualifying configs, {len(ok)} with >=20 validation trades")

    # ---------------- does the search ranking generalize? ----------------
    print("\n=== DOES THE SEARCH RANKING GENERALIZE OUT OF SAMPLE? ===")
    for label, sc, vc in [
        ("win rate", "search_win_rate", "val_win_rate"),
        ("profit factor", "search_profit_factor", "val_pf"),
        ("total pnl", "search_total_pnl", "val_pnl"),
    ]:
        r, p = rho(ok[sc].values, ok[vc].values)
        verdict = "GENERALIZES" if (p < 0.05 and r > 0) else "NO / NEGATIVE"
        print(f"  spearman(search {label:13s}, validation {label:13s}) = "
              f"{r:+.4f}  p={p:.3g}   -> {verdict}")

    # ---------------- decay of the top-ranked configs ----------------
    print("\n=== DECAY: top configs by SEARCH metric, measured on validation ===")
    for topn in (10, 25, 100):
        t_wr = ok.nlargest(topn, "search_win_rate")
        t_pf = ok.nlargest(topn, "search_profit_factor")
        print(f"  top {topn:3d} by search win rate : search={t_wr['search_win_rate'].mean():.4f} "
              f"-> validation={t_wr['val_win_rate'].mean():.4f} "
              f"(decay {t_wr['val_win_rate'].mean()-t_wr['search_win_rate'].mean():+.4f})")
        print(f"  top {topn:3d} by search PF       : search={t_pf['search_profit_factor'].mean():.4f} "
              f"-> validation={t_pf['val_pf'].replace([np.inf],np.nan).mean():.4f} "
              f"(decay {t_pf['val_pf'].replace([np.inf],np.nan).mean()-t_pf['search_profit_factor'].mean():+.4f})")

    print(f"\n  validation win rate across ALL configs: mean={ok['val_win_rate'].mean():.4f} "
          f"median={ok['val_win_rate'].median():.4f}")
    print(f"  validation PF       across ALL configs: mean="
          f"{ok['val_pf'].replace([np.inf],np.nan).mean():.4f} "
          f"median={ok['val_pf'].replace([np.inf],np.nan).median():.4f}")

    # ---------------- profitable in BOTH periods: better than chance? ----------------
    print("\n=== PROFITABLE IN BOTH PERIODS -- MORE THAN CHANCE? ===")
    p_search = (ok["search_profit_factor"] > 1).mean()
    p_valid = (ok["val_pf"] > 1).mean()
    both = ((ok["search_profit_factor"] > 1) & (ok["val_pf"] > 1)).sum()
    expected_if_independent = p_search * p_valid * len(ok)
    print(f"  profitable in search    : {p_search*100:.2f}%")
    print(f"  profitable in validation: {p_valid*100:.2f}%")
    print(f"  profitable in BOTH      : {both} configs")
    print(f"  expected by chance if the two were independent: {expected_if_independent:.1f}")
    if expected_if_independent > 0:
        bt = binomtest(int(both), len(ok), p_search * p_valid, alternative="greater")
        print(f"  binomial test p = {bt.pvalue:.4g}  -> "
              f"{'MORE than chance' if bt.pvalue < 0.05 else 'INDISTINGUISHABLE FROM CHANCE'}")

    # ---------------- what you'd actually have deployed ----------------
    print("\n=== WHAT YOU WOULD HAVE ACTUALLY DEPLOYED ===")
    cols = ["rsi_len", "stoch_len", "adx_len", "adx_th", "os_th", "k_th", "d_th",
            "fixed_sl", "trail_sl", "gain_th", "search_win_rate", "search_profit_factor",
            "search_total_pnl", "val_win_rate", "val_pf", "val_pnl", "val_n_trades", "val_freq"]
    print("\n-- picked by best SEARCH win rate --")
    print(ok.nlargest(1, "search_win_rate")[cols].to_string(index=False))
    print("\n-- picked by best SEARCH profit factor --")
    print(ok.nlargest(1, "search_profit_factor")[cols].to_string(index=False))
    print("\n-- (for reference only, unknowable in advance) best VALIDATION PF --")
    print(ok.nlargest(1, "val_pf")[cols].to_string(index=False))

    ok.to_csv("winrate_analysis_final.csv", index=False)


if __name__ == "__main__":
    main()
