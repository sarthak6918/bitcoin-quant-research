"""Join funding rate + OI metrics + order-flow features onto the existing
hourly perp OHLCV frame (11_Spot_vs_Perp_Analysis/perp_hourly.csv).

Funding is 8-hourly -> forward-filled onto the hourly index (a funding print
at 00:00 is known and in effect until the next print at 08:00 -- forward-fill
is the causal, no-lookahead direction here, not interpolation).
OI metrics and order-flow are already hourly, aligned by left-join.

Output: btc_perp_hourly_with_microstructure.csv
"""
import pandas as pd

PERP = "../11_Spot_vs_Perp_Analysis/perp_hourly.csv"
SPOT = "../11_Spot_vs_Perp_Analysis/spot_hourly.csv"
FUNDING = "funding_rate_full_history.csv"
OI = "oi_metrics_hourly.csv"
FLOW = "orderflow_hourly_features.csv"
OUT = "btc_perp_hourly_with_microstructure.csv"


def main():
    perp = pd.read_csv(PERP, parse_dates=["timestamp"]).set_index("timestamp")
    spot = pd.read_csv(SPOT, parse_dates=["timestamp"]).set_index("timestamp")
    perp = perp.join(spot[["close"]].rename(columns={"close": "spot_close"}), how="left")
    perp["basis_pct"] = perp["close"] / perp["spot_close"] - 1.0

    funding = pd.read_csv(FUNDING, parse_dates=["fundingTime"]).set_index("fundingTime")
    funding_hourly = funding[["fundingRate"]].resample("1h").ffill()
    perp = perp.join(funding_hourly, how="left")
    perp["fundingRate"] = perp["fundingRate"].ffill()

    try:
        oi = pd.read_csv(OI, parse_dates=["timestamp"]).set_index("timestamp")
        perp = perp.join(oi, how="left")
    except FileNotFoundError:
        print(f"WARNING: {OI} not found yet -- skipping OI join. Run fetch_oi_metrics.py first.")

    try:
        flow = pd.read_csv(FLOW, parse_dates=["timestamp"]).set_index("timestamp")
        flow = flow.drop(columns=["price_close"], errors="ignore")
        perp = perp.join(flow, how="left")
    except FileNotFoundError:
        print(f"WARNING: {FLOW} not found yet -- skipping order-flow join. Run build_orderflow_features.py first.")

    perp = perp.reset_index()
    perp.to_csv(OUT, index=False)
    n_with_flow = perp["ofi"].notna().sum() if "ofi" in perp else 0
    print(f"Wrote {len(perp)} rows -> {OUT}")
    print(f"  rows with order-flow features populated: {n_with_flow}")
    print(f"  date range: {perp['timestamp'].min()} -> {perp['timestamp'].max()}")


if __name__ == "__main__":
    main()
