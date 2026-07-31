# 03 — CatBoost Model A: Deployed Baseline (10-bar barrier)

This is the model that was genuinely live in production before any HMM
regime work started — confirmed directly from the actual source code
(`retrain_from_signals_all_signals.py`, `model_monitor.py`), both of which
train on `binary_target`, the 10-bar vertical-barrier label.

## Files

| File | What it is |
|---|---|
| `train_catboost_model.py` | Standalone script — reproduces this model exactly. Re-verified in a clean directory before packaging: reproduces AUC 0.8575 ± 0.0027 bit-for-bit |
| `model_config.json` | Full hyperparameters, feature list, target definition, results, and the known feature-basis bug writeup |
| `trained_models/model_seed{42-46}.cbm` | The 5 trained model files (averaged at inference) |
| `training_data/train_split.csv` | 3,446 rows, 2017-08-21 → 2023-04-17 |
| `training_data/validation_split.csv` | 737 rows, 2023-04-18 → 2024-08-23 (early stopping only) |
| `training_data/test_split_frozen_holdout.csv` | 738 rows, 2024-08-23 → 2025-12-07 (permanent, never touched during training) |

## Key facts

- **Target**: `binary_target` — 10-bar vertical timeout, triple barrier at max(2×ATR%, 1.0%)
- **Features**: 27 (no HMM/regime input)
- **Frozen holdout AUC**: 0.8575 ± 0.0027
- **Fresh 2026 live-data AUC**: ~0.55–0.57 (see section 05/06 for the full, corrected investigation)

## Known issue, disclosed transparently

4 of the 27 features (`atr_pct`, `ema9_dist`, `keltner_pos`,
`supertrend_dist`) were built on `corrected_entry_price` (next-bar open) in
this training data, but the live bot computes these same features off the
raw signal-candle close. Confirmed and fixed directly — see
`09_Bugs_Found_And_Fixed`. Fixing it did NOT resolve the live AUC gap on
its own, but it's a real mismatch worth patching in the live codebase
regardless.

## Reproduction

```bash
python train_catboost_model.py
# expect: Mean frozen-holdout AUC: 0.8575 ± 0.0027
```
