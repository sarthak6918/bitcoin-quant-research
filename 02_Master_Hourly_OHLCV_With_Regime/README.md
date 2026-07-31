# 02 — Master Hourly OHLCV With Regime

The single most-reused dataset in this entire project: continuous 1-hour
BTC/USDT OHLCV, matched to Binance's own candle grid, with the HMM regime
(filtered/causal probabilities) forward-filled onto every hourly bar.

## Files

| File | What it is |
|---|---|
| `btc_1h_ohlcv_2017_2026_with_regime.csv` | 78,150 hourly bars, 2017-08-17 04:00 → 2026-07-22 17:00. Columns: `timestamp, open, high, low, close, volume, filtered_state, filtered_state_label, filtered_prob_state0..4` |
| `fetch_and_map_script.py` | Script used to fetch this data and map the daily HMM regime onto it |

## Why this file matters

Every feature-rebuild, every triple-barrier label, every regime merge in
this whole project traces back to this one file. It's the ground-truth
price series everything else was checked against — including finding the
IST-timezone bug and the `corrected_entry_price` feature-basis bug (see
`09_Bugs_Found_And_Fixed`), both of which were caught by comparing external
signal files' prices against this dataset.

## Regime mapping — no lookahead

The daily HMM regime is forward-filled onto hourly bars with a **1-day
lag**: a day's regime is only "available" starting the following day at
00:00, since it isn't known until that day's daily bar has actually
closed. Confirmed via QA: only 356 unlabeled rows at the very start
(feature warmup), 128 missing hours total (0.16%, normal exchange
downtime), zero duplicate timestamps.

## Data quality confirmed

- 0 duplicate timestamps
- 0.16% missing hours (exchange downtime, not a bug)
- Regime state distribution matches the daily model's own distribution
  within expected rounding from hour-weighting vs day-weighting
