# 10 — Direction Prediction Rebuild (Honest Result: 0.7 AUC Target NOT Met)

This folder is a fresh, independent rebuild targeting the reframed goal:
predict the **direction** of price N bars forward (not "does this specific
triple-barrier trade win or lose," which is what folders 03-06 modeled).
Built entirely from `02_Master_Hourly_OHLCV_With_Regime` — every one of the
78,150 hourly bars is used as a training row, not just the ~4,100 rows where
the StochRSI+ADX entry trigger fired. This gives far more statistical power
and removes stop-loss/path noise from the label.

## Files

| File | What it is |
|---|---|
| `build_features.py` | Causal feature engineering: 60+ technical/multi-timeframe/regime features, each provably using only data at/before bar t |
| `leakage_audit.py` | Mechanical leakage test — corrupts all OHLCV after a cutoff row and verifies every feature value before the cutoff is unchanged. **Passed: 0 leaking columns**, both before and after adding multi-timeframe features |
| `features_labeled.csv` | Output of `build_features.py` — features + 7 horizon labels (1,2,3,5,10,20,50 bars) |
| `train_walkforward.py` | Purged + embargoed expanding-window walk-forward CV (5 folds, training pool only, pre-2025) |
| `final_holdout_eval.py` | Single-touch evaluation on 2025-01-01 → 2026-07-22, never used during model/horizon selection |
| `final_holdout_predictions.csv` | Per-bar predictions on the final holdout |

## Method

- **Label**: `sign(close[t+n] - close[t])`, n ∈ {1,2,3,5,10,20,50} hourly bars
- **Features**: multi-lookback returns/vol/skew/kurt, RSI/StochRSI, ADX/DI,
  ATR%, EMA distance+slope, Keltner/Bollinger position, Supertrend, OBV,
  range structure, return autocorrelation, hour-of-day/day-of-week, HMM
  regime probabilities (causal forward-algorithm decode from folder 01),
  **plus 4h and daily trend context** — built from fully-completed
  higher-timeframe bars only, shifted by 1 and forward-filled, so the
  current hourly row never sees its own still-forming daily/4h bar
- **Leakage check**: mechanical, not visual — corrupt-the-future test on
  every feature column, passes cleanly
- **Validation**: chronological, purged (drop training rows whose label
  window overlaps the test fold start) + embargoed (2n-bar gap at every
  fold boundary), 5 expanding folds, 5 seeds averaged, 180-day recency
  half-life sample weighting — same discipline as the rest of this project
- **Final holdout**: 2025-01-01 → 2026-07-22 (13,625 hourly bars), touched
  exactly once, after the horizon was already selected from walk-forward

## Results

```
Walk-forward AUC by horizon (training pool, pre-2025, purged/embargoed):
  n=1  bar : 0.5722 +/- 0.0031   <- best, most stable
  n=2  bars: 0.5668 +/- 0.0095
  n=3  bars: 0.5620 +/- 0.0098
  n=5  bars: 0.5580 +/- 0.0092
  n=10 bars: 0.5424 +/- 0.0153
  n=20 bars: 0.5465 +/- 0.0179
  n=50 bars: 0.5397 +/- 0.0308

Final holdout AUC (n=1, 2025-01 -> 2026-07-22): 0.5382
Monthly range: 0.51 - 0.57, no cliff, no single lucky month
```

## Honest conclusion: the 0.7 AUC target was not reached, and is very
## unlikely reachable with this data

This is the **third independent confirmation** in this project (after
CatBoost/MLP/LSTM on the triple-barrier label in folders 03-07) that
price/volume-derived technical features on liquid BTC/USDT hourly data cap
out around **AUC 0.53-0.57**, regardless of:
- label formulation (triple-barrier win/loss *or* plain forward direction)
- how much data is used (4,100 signal rows *or* all 78,150 bars)
- horizon (1 to 50 bars)
- model architecture (gradient boosting, MLP, LSTM)
- feature richness (adding 4h/daily multi-timeframe context moved n=1 AUC
  from ~0.55 to 0.57 — a real, small improvement, not noise, but nowhere
  near 0.7)

This is not a bug or a data problem — the leakage audit is mechanical and
passed, the splits are purged/embargoed, and the result is stable across
20 consecutive out-of-sample months with no single outlier driving it. It
is consistent with the honest market-structure explanation: BTC/USDT is
one of the most liquid, most heavily arbitraged instruments in crypto, and
public, univariate, price-derived technical indicators are exactly the
kind of information that gets arbitraged out first. An AUC of 0.7 on
hourly direction would imply an extraordinarily large, persistent,
publicly-visible inefficiency — not impossible in principle, but not
supported by three independent architectures and two label formulations
all landing in the same 0.53-0.57 band.

**What would plausibly move this toward 0.7**, none of which is in the
current data:
1. **Order flow / order book imbalance** (bid-ask depth, aggressor volume) —
   the single most likely lever; this data isn't in any file provided
2. **Funding rate + open interest** (perp positioning) — explicitly
   descoped early in this project per `00_METHODOLOGY_AND_FINDINGS.md`
3. **Cross-asset leading signals** (e.g., short-horizon lead/lag from
   ETH, DXY, equity futures) — not in the current dataset
4. **Much longer horizons** (multi-day/weekly) where macro/trend factors
   dominate noise more — worth testing if the trading use case tolerates it
5. **A different, less-efficient instrument** than BTC/USDT spot

**What will NOT get you to 0.7 honestly**: more feature engineering on
this same OHLCV data, larger models, or more hyperparameter search — this
project has now tried all three, twice, independently, and hit the same
ceiling both times. Continuing to iterate on this exact input space would
mean either self-deceiving via subtle leakage or overfitting the holdout
by repeated peeking — both of which would produce a number that looks
like 0.7 in backtest and fails live, which is the exact failure mode this
whole project was launched to diagnose and fix.
