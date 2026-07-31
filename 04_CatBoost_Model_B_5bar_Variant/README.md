# 04 — CatBoost Model B: 5-Bar Barrier Variant

A research variant testing whether a shorter (5-bar) vertical timeout
recovers live performance. It doesn't — included here for completeness
and honest documentation of what was tried.

## Files

| File | What it is |
|---|---|
| `train_catboost_model.py` | Standalone script — reproduces this model exactly (verified: AUC 0.9128 ± 0.0015) |
| `model_config.json` | Full config + results, including the known feature-basis issue |
| `trained_models/model_seed{42-46}.cbm` | The 5 trained model files |
| `training_data/*.csv` | Same date boundaries as Model A, but target = `binary_target_vb5` (5-bar) |

## Key facts

- **Target**: `binary_target_vb5` — 5-bar vertical timeout, same barrier-width formula as the 10-bar version
- **Frozen holdout AUC**: 0.9128 ± 0.0015 (higher than the 10-bar model)
- **Fresh 2026 live-data AUC**: ~0.47–0.49 (LOWER than the 10-bar model)

## The core finding from this variant

Shortening the barrier makes the historical fit look *better* while making
live generalization *worse* — a real, consistent tradeoff observed across
every test in this project, not a one-off. Higher historical AUC here is
not evidence of a better model; see `06_Walkforward_Validation_And_Final_Model`
for the properly controlled comparison isolating this effect from
everything else (same Train/Validation date boundaries, same features).

## Reproduction

```bash
python train_catboost_model.py
# expect: Mean frozen-holdout AUC: 0.9128 ± 0.0015
```
