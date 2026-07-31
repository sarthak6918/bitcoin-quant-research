# 09 — Bugs Found And Fixed

Three genuine, confirmed bugs were found during this investigation. All
three are documented here with the exact evidence, not just asserted.

## Files

| File | What it is |
|---|---|
| `reverse_engineer_feature_formulas.py` | Script that reverse-engineered and confirmed the exact triple-barrier and feature formulas by testing candidates against the historical dataset |
| `corrected_direction_relabeling_script.py` | Script that relabels trades using the CORRECT direction convention (see Bug 3) |
| `feature_basis_bug_fixed_dataset.csv` | The historical dataset with the 4 affected features (`atr_pct`, `ema9_dist`, `keltner_pos`, `supertrend_dist`) recomputed on the correct raw-price basis |

## Bug 1: IST timezone artifact in TradingView exports

Every timestamp in `bar10_training_dataset_FULLY_CORRECTED.csv` (4,921
rows) and in later TradingView strategy exports landed on the `:30`
minute mark — not a coincidence at that consistency. Confirmed by testing
a −5:30 shift against the independently-fetched UTC OHLCV: mean price
alignment error dropped from ~1.2% (unshifted) to ~0.05–0.48% (shifted),
depending on the specific export. Root cause: TradingView renders trade-
list exports in the account's local timezone (IST, confirmed via INR
currency columns on later files), not UTC.

## Bug 2: Feature-basis mismatch (`corrected_entry_price` vs. raw price)

`atr_pct`, `ema9_dist`, `keltner_pos`, and `supertrend_dist` in the
original historical training set were all computed relative to
`corrected_entry_price` (the next bar's open — a valid fix for a *label*
lookahead bug), but the live bot (`signal_generator_v3.py` +
`video_features.py`) computes these same features relative to the raw
signal-candle close. Confirmed via an exact-match test: the discrepancy
between the two price bases matches the magnitude of the entry-price
correction itself to within floating-point precision, across every
affected column. **Tested directly**: recomputing features on the correct
(raw) basis and retraining did NOT resolve the live AUC gap (historical
Test AUC 0.858→0.848, fresh-forward AUC 0.566→0.557) — a real bug worth
fixing in the live codebase, but not the primary driver of the gap.

## Bug 3: Direction convention (`execution_signal` vs. `Signal`)

The production labeling code (`retrain_from_signals_all_signals.py`) uses
`label_side = raw_signal if raw_signal in {BUY,SELL} else exec_sig` — since
`raw_signal` (the `Signal` column) is always BUY/SELL, the label ALWAYS
uses the raw technical trigger's direction, never `execution_signal` (what
the ML model decided to do — take as-is, reverse, or skip). An earlier
pass in this analysis used `execution_signal` for some live-trade
relabeling, which silently flipped the outcome for every reversed
(Bucket E) trade. Caught and fixed — affected 29 of 66 trades in one
specific intermediate file; does not affect any of the final, packaged
datasets in this zip.

## Why none of these three explain the core live-AUC gap

Each was tested directly, not just fixed and assumed to help:
- Timezone fix: improved price alignment, didn't change AUC materially
- Feature-basis fix: no change in historical or live AUC
- Direction fix: corrected specific trade outcomes, but the aggregate
  finding (near-random AUC on genuinely fresh data) held before and after

The real answer — see the master methodology doc — is that a from-scratch
rebuild with all three bugs fixed still shows ~0.53 walk-forward AUC even
on purely historical data, meaning the *original* dataset's high AUC
(0.85-0.91) was itself the artifact, not a real signal that later
"decayed."
