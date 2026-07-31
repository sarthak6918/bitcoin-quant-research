"""Generate chart PNGs (base64-embeddable) for the spot vs perp report."""
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3de"

plt.rcParams.update({
    "font.size": 11, "text.color": TEXT, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})


def to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def chart_price(spot, perp):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(spot["timestamp"], spot["close"], color=BLUE, lw=1.3, label="Spot")
    ax.plot(perp["timestamp"], perp["close"], color=ORANGE, lw=1.3, label="Perp futures")
    ax.set_yscale("log")
    ax.set_ylabel("Price (USDT, log scale)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("BTC/USDT Spot vs Perpetual Futures — Daily Close, 2019–2026", loc="left", fontsize=12)
    return to_b64(fig)


def chart_volatility(spot, perp, window=30):
    r_s = np.log(spot["close"] / spot["close"].shift(1))
    r_p = np.log(perp["close"] / perp["close"].shift(1))
    vol_s = r_s.rolling(window).std() * np.sqrt(365) * 100
    vol_p = r_p.rolling(window).std() * np.sqrt(365) * 100

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(spot["timestamp"], vol_s, color=BLUE, lw=1.1, label="Spot")
    ax.plot(perp["timestamp"], vol_p, color=ORANGE, lw=1.1, label="Perp futures", alpha=0.85)
    ax.set_ylabel("Annualized realized vol (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"{window}-Day Rolling Realized Volatility — Spot vs Perp", loc="left", fontsize=12)
    return to_b64(fig)


def chart_acf(summary, kind, title, fname_note):
    lags = list(range(1, 11))
    spot_row = summary[summary["label"] == "spot_daily"].iloc[0]
    perp_row = summary[summary["label"] == "perp_daily"].iloc[0]
    spot_vals = [spot_row[f"acf_{kind}_lag{l}"] for l in lags]
    perp_vals = [perp_row[f"acf_{kind}_lag{l}"] for l in lags]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(lags))
    w = 0.36
    ax.bar(x - w/2, spot_vals, width=w, color=BLUE, label="Spot")
    ax.bar(x + w/2, perp_vals, width=w, color=ORANGE, label="Perp futures")
    ax.axhline(0, color=TEXT, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"lag {l}" for l in lags])
    ax.set_ylabel("Autocorrelation")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(title, loc="left", fontsize=12)
    return to_b64(fig)


def chart_volume(spot, perp):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(spot["timestamp"], spot["volume"], color=BLUE, lw=1.0, label="Spot (BTC)", alpha=0.85)
    ax.plot(perp["timestamp"], perp["volume"], color=ORANGE, lw=1.0, label="Perp futures (BTC)", alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("Daily volume (BTC, log scale)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Daily Traded Volume — Spot vs Perp Futures", loc="left", fontsize=12)
    return to_b64(fig)


def main():
    spot = pd.read_csv("spot_daily.csv", parse_dates=["timestamp"])
    perp = pd.read_csv("perp_daily.csv", parse_dates=["timestamp"])
    summary = pd.read_csv("summary_stats.csv")

    imgs = {
        "price": chart_price(spot, perp),
        "volatility": chart_volatility(spot, perp),
        "acf_return": chart_acf(summary, "return", "Return Autocorrelation (Daily) — Weak Mean-Reversion, Both Series", ""),
        "acf_vol_cluster": chart_acf(summary, "abs_return", "Volatility Clustering: |Return| Autocorrelation (Daily) — the ARCH Effect", ""),
        "volume": chart_volume(spot, perp),
    }
    import json
    with open("chart_images.json", "w") as f:
        json.dump(imgs, f)
    print("saved chart_images.json with", len(imgs), "charts")


if __name__ == "__main__":
    main()
