# 05 — Final Rigorous Dataset: 4,112 Signals, 2017–2026, Built From Scratch

**This is the most trustworthy dataset in the entire project.** Unlike the
earlier `bar10_training_dataset_FULLY_CORRECTED.csv` (which had at least
two confirmed bugs — see `09_Bugs_Found_And_Fixed`), every single value in
this file was computed independently from `02_Master_Hourly_OHLCV_With_Regime`,
using the exact production feature code, with no inherited assumptions
from any prior processing step.

## File

| File | What it is |
|---|---|
| `master_dataset_4123_signals_2017_2026.csv` | 4,112 signals (of 4,123 raw entries — 11 dropped for insufficient history), 2017-09-01 → 2026-07-21 |
| `video_features_production_code.py` | The exact, unmodified production module used to compute the 7 EMA/Keltner/Supertrend/log-return features (confirmed lookahead-free via its own self-test) |

## How it was built (every step verified, not assumed)

1. **Source**: a TradingView strategy export (`Gold_MCX_GOLD1___Final__BINANCE_BTCUSDT_2026-07-27.csv`,
   not included here — raw upload), containing 4,123 entry signals from an
   India-based TradingView account (confirmed via INR currency columns).
2. **Timezone correction**: all timestamps were in IST, confirmed empirically
   by testing candidate shifts against this project's own UTC OHLCV —
   the −5:30 shift brought mean price alignment error down to 0.0007%
   (essentially perfect) across all 4,123 signals.
3. **Features**: all 27 base features + 5 HMM regime probabilities rebuilt
   from scratch using the exact `ta`-library methods from `signal_generator_v3.py`
   and the unmodified `video_features.py` module — matching the live bot's
   computation exactly, not the training-set's original (buggy) basis.
4. **Labels**: 5-bar triple barrier (barrier = max(2×ATR%, 1.0%)), direction
   from the raw `Signal` column (not `execution_signal` — see bug #3 in
   `09_Bugs_Found_And_Fixed`), computed via bar-by-bar path scanning against
   the continuous OHLCV.

## Confirmed data quality

- 0 NaN values anywhere in the feature/label columns
- Barrier hit distribution: 2,931 vertical / 664 lower / 515 upper / 2 ambiguous
- Overall win rate: 48.69%
- 0 unmatched regime rows

## The headline finding from this dataset

Walk-forward validation (5 expanding-window folds, 2019–2024) on this
cleanly-built data gives **AUC = 0.532 ± 0.022** (verified directly against
this exact file before packaging) — barely above random, even on purely
historical data with no live-data involvement at all. This is the critical
finding: the earlier ~0.85–0.91 historical AUC figures (Models A and B)
were very likely inflated by artifacts in how that older dataset was
built, not genuine predictive signal. See
`06_Walkforward_Validation_And_Final_Model` for the full breakdown.
