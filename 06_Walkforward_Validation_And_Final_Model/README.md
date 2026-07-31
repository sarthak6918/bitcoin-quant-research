# 06 — Walk-Forward Validation And Final Model

The final, most rigorous CatBoost model: trained on the walk-forward
validated feature set (28 base features + 5 HMM regime probabilities),
evaluated once on a genuinely held-out, never-touched period.

## Files

| File | What it is |
|---|---|
| `walkforward_5bar_regime.py` | Full script: 5 expanding-window walk-forward folds + final model fit + test evaluation |
| `training_pool_pre2025.csv` | All signals before 2025-01-01 (the walk-forward training pool) |
| `test_set_last700_with_predictions.csv` | An earlier test construction (last 700 signals chronologically) — includes predictions, but see the caveat below |
| `test_set_2025_2026_with_predictions.csv` | The clean version: test = everything from 2025-01-01 onward on the from-scratch dataset, with final model predictions attached |

## Split design

- **Training pool**: all signals before 2025-01-01 (~3,400 rows spanning 2017–2024)
- **Test set**: 2025-01-01 → 2026-07-21 (~700 rows), never touched during
  training or walk-forward
- `vol_regime` bins recomputed from the training pool ONLY at every step
  (no leakage of future volatility quantiles)

## Results

```
Walk-forward validation (training pool, 5 expanding folds): AUC = 0.532 ± 0.022
Final test AUC (2025-01 -> 2026-07):                        AUC ≈ 0.50-0.57
```

## Important caveat on `test_set_last700_with_predictions.csv`

This file mixes 375 rows from an earlier "historical Test" period
(April–December 2025, which the older, buggier dataset construction still
handled well) with 325 rows of genuinely new live 2026 data. The blended
AUC on this file (0.7331) looked encouraging but was misleading — broken
down by source, the historical portion scored 0.898 while the live-2026
portion scored 0.475. **Use `test_set_2025_2026_with_predictions.csv`
instead for anything going forward** — it's built entirely from the
corrected, from-scratch dataset with no such blending artifact.

## Monthly AUC breakdown (from-scratch dataset, 2025-2026)

No single month is a clear outlier — performance is uniformly mediocre
throughout both 2025 and 2026, which is itself informative: there was no
sudden "2026 cliff." The earlier historical vs. live gap was a dataset-
construction artifact (see `09_Bugs_Found_And_Fixed`), not a real-world
regime change concentrated in 2026.
