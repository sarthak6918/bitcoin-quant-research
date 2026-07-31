"""
Spot vs. Perpetual Futures statistical comparison, BTC/USDT, 2019-2026.

Computes, for both series, at daily and hourly granularity:
  - mean and std of log returns
  - autocorrelation of returns (lag 1-10) -- tests the random-walk hypothesis
  - volatility clustering: autocorrelation of |returns| and returns^2
    (the "ARCH effect" -- large moves cluster in time even when returns
    themselves don't autocorrelate)
  - volume: mean, std, coefficient of variation, trend
Also a year-by-year breakdown (2019-2026) for both series.
"""
import numpy as np
import pandas as pd

YEARS = list(range(2019, 2027))


def log_returns(close):
    return np.log(close / close.shift(1)).dropna()


def autocorr(x, lag):
    x = np.asarray(x, dtype=float)
    if len(x) <= lag:
        return np.nan
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]


def ljung_box_stat(x, lags):
    """Simple Ljung-Box Q statistic (no external dependency)."""
    n = len(x)
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    acf = [autocorr(x, k) for k in range(1, lags + 1)]
    q = n * (n + 2) * sum((a ** 2) / (n - k) for k, a in enumerate(acf, start=1) if np.isfinite(a))
    return q


def series_stats(df, label, ret_lags=10):
    c = df["close"]
    r = log_returns(c)
    abs_r = r.abs()
    sq_r = r ** 2
    vol_series = df["volume"]

    stats = {
        "label": label,
        "n_obs": len(df),
        "start": df["timestamp"].min(),
        "end": df["timestamp"].max(),
        "mean_log_return": r.mean(),
        "std_log_return": r.std(),
        "annualization_factor_note": "not annualized -- per-bar units",
        "skew": r.skew(),
        "kurtosis_excess": r.kurtosis(),
        "min_return": r.min(),
        "max_return": r.max(),
    }
    for lag in range(1, ret_lags + 1):
        stats[f"acf_return_lag{lag}"] = autocorr(r.values, lag)
    for lag in range(1, ret_lags + 1):
        stats[f"acf_abs_return_lag{lag}"] = autocorr(abs_r.values, lag)
        stats[f"acf_sq_return_lag{lag}"] = autocorr(sq_r.values, lag)
    stats["ljung_box_Q_returns_10lag"] = ljung_box_stat(r.values, 10)
    stats["ljung_box_Q_abs_returns_10lag"] = ljung_box_stat(abs_r.values, 10)

    stats["volume_mean"] = vol_series.mean()
    stats["volume_std"] = vol_series.std()
    stats["volume_cv"] = vol_series.std() / vol_series.mean()
    stats["volume_median"] = vol_series.median()
    # simple linear trend in volume, in units of volume/day
    t = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 86400
    slope = np.polyfit(t, vol_series.values, 1)[0]
    stats["volume_trend_per_day"] = slope
    return stats, r, abs_r


def yearly_breakdown(df, label):
    rows = []
    for y in YEARS:
        sub = df[df["timestamp"].dt.year == y]
        if len(sub) < 30:
            continue
        r = log_returns(sub["close"])
        rows.append(dict(
            series=label, year=y, n=len(sub),
            mean_log_return=r.mean(), std_log_return=r.std(),
            acf_return_lag1=autocorr(r.values, 1),
            acf_abs_return_lag1=autocorr(r.abs().values, 1),
            acf_sq_return_lag1=autocorr((r**2).values, 1),
            volume_mean=sub["volume"].mean(),
        ))
    return pd.DataFrame(rows)


def main():
    spot_d = pd.read_csv("spot_daily.csv", parse_dates=["timestamp"])
    perp_d = pd.read_csv("perp_daily.csv", parse_dates=["timestamp"])
    spot_h = pd.read_csv("spot_hourly.csv", parse_dates=["timestamp"])
    perp_h = pd.read_csv("perp_hourly.csv", parse_dates=["timestamp"])

    all_stats = []
    series_returns = {}
    for df, label in [(spot_d, "spot_daily"), (perp_d, "perp_daily"),
                       (spot_h, "spot_hourly"), (perp_h, "perp_hourly")]:
        s, r, abs_r = series_stats(df, label)
        all_stats.append(s)
        series_returns[label] = (r, abs_r)
        print(f"{label}: n={s['n_obs']}  mean={s['mean_log_return']:.6f}  "
              f"std={s['std_log_return']:.6f}  acf_r1={s['acf_return_lag1']:.4f}  "
              f"acf_|r|1={s['acf_abs_return_lag1']:.4f}  vol_mean={s['volume_mean']:.1f}")

    summary = pd.DataFrame(all_stats)
    summary.to_csv("summary_stats.csv", index=False)

    yearly = pd.concat([
        yearly_breakdown(spot_d, "spot"),
        yearly_breakdown(perp_d, "perp"),
    ], ignore_index=True)
    yearly.to_csv("yearly_breakdown.csv", index=False)

    # aligned overlap period (both series exist): 2019-09-08 onward
    overlap_start = max(spot_d["timestamp"].min(), perp_d["timestamp"].min())
    spot_o = spot_d[spot_d["timestamp"] >= overlap_start].reset_index(drop=True)
    perp_o = perp_d[perp_d["timestamp"] >= overlap_start].reset_index(drop=True)
    merged = spot_o[["timestamp", "close"]].merge(
        perp_o[["timestamp", "close"]], on="timestamp", suffixes=("_spot", "_perp"))
    r_spot = log_returns(merged["close_spot"])
    r_perp = log_returns(merged["close_perp"])
    corr_returns = np.corrcoef(r_spot.values, r_perp.values)[0, 1]
    price_diff_pct = ((merged["close_perp"] - merged["close_spot"]) / merged["close_spot"]).abs()
    print(f"\noverlap period ({overlap_start.date()} onward, n={len(merged)}):")
    print(f"  correlation of daily returns spot vs perp: {corr_returns:.6f}")
    print(f"  mean abs basis (perp-spot)/spot: {price_diff_pct.mean()*100:.4f}%  "
          f"max: {price_diff_pct.max()*100:.4f}%")

    with open("overlap_stats.txt", "w") as f:
        f.write(f"overlap_start={overlap_start}\nn={len(merged)}\n"
               f"corr_returns={corr_returns}\nmean_abs_basis_pct={price_diff_pct.mean()*100}\n"
               f"max_abs_basis_pct={price_diff_pct.max()*100}\n")

    print("\nsaved summary_stats.csv, yearly_breakdown.csv, overlap_stats.txt")


if __name__ == "__main__":
    main()
