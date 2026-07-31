"""Quick, honest first read: does any of this free-tier data actually carry
forward-return information, before spending a cent on paid order-book depth?

Two checks, matching the two live hypotheses from the research memo:

1. Basis reversion (Strategy 1) -- does basis_z predict forward basis change?
2. Order-flow / OI information content (Strategies 2-3) -- Spearman IC of
   each microstructure feature against forward returns at several horizons.

This is a diagnostic, not a backtest: no train/test split, no costs, no
purging. A feature that shows nothing here is very unlikely to survive a
rigorous pipeline; a feature that shows something here still has to survive
one before it's a strategy. Treat positive results as "worth the AFML
pipeline", not as an edge.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

IN = "btc_perp_hourly_with_microstructure.csv"
HORIZONS = [1, 4, 24]


def ic(x, y):
    mask = x.notna() & y.notna()
    if mask.sum() < 100:
        return np.nan, mask.sum()
    rho, p = spearmanr(x[mask], y[mask])
    return rho, mask.sum()


def main():
    df = pd.read_csv(IN, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded {len(df)} hourly rows, {df['timestamp'].min()} -> {df['timestamp'].max()}\n")

    # ---- Check 1: basis reversion ----
    print("=" * 70)
    print("CHECK 1 -- Basis reversion (Strategy 1)")
    print("=" * 70)
    df["basis_z"] = (df["basis_pct"] - df["basis_pct"].rolling(24 * 7).mean()) / df["basis_pct"].rolling(24 * 7).std()
    for h in HORIZONS:
        fwd_basis_change = df["basis_pct"].shift(-h) - df["basis_pct"]
        rho, n = ic(df["basis_z"], fwd_basis_change)
        print(f"  basis_z -> {h:>3}h forward basis change:  IC = {rho:+.4f}  (n={n})")
    print("  Expect NEGATIVE IC if reversion is real (high z -> basis falls back).")

    # ---- Check 2: order-flow / OI information content ----
    print()
    print("=" * 70)
    print("CHECK 2 -- Order-flow & OI feature IC vs forward BTC returns")
    print("=" * 70)
    df["fwd_ret"] = {}
    candidate_features = [
        "ofi", "taker_buy_ratio", "net_taker_flow", "large_trade_net_flow",
        "trade_count_imbalance", "avg_trade_size",
        "sum_open_interest", "sum_toptrader_long_short_ratio",
        "sum_taker_long_short_vol_ratio", "fundingRate",
    ]
    available = [c for c in candidate_features if c in df.columns]
    missing = [c for c in candidate_features if c not in df.columns]
    if missing:
        print(f"  (not yet joined, skipping: {missing})")

    for feat in available:
        row = []
        for h in HORIZONS:
            fwd_ret = np.log(df["close"].shift(-h) / df["close"])
            rho, n = ic(df[feat], fwd_ret)
            row.append(f"{h:>2}h: {rho:+.4f} (n={n})")
        print(f"  {feat:<32} " + "   ".join(row))

    print()
    print("Reading this: |IC| > ~0.02-0.03 with large n is worth pursuing further")
    print("in the full purged/embargoed pipeline. Anything near 0 across all")
    print("horizons is the same wall this project has hit before -- don't force it.")


if __name__ == "__main__":
    main()
