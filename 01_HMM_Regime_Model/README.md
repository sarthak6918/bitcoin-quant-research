# 01 — HMM Regime Model

A Gaussian Hidden Markov Model that classifies each day of BTC/USDT history
into one of 5 latent market regimes, discovered statistically rather than
via hand-picked rule thresholds.

## Files

| File | What it is |
|---|---|
| `train_hmm_model.py` | Full training pipeline: fetches daily OHLCV, builds features, fits the HMM across K=2..7 with health checks, decodes states, saves everything |
| `hmm_model_config.json` | Exact hyperparameters, feature definitions, per-state statistics, transition matrix — pulled directly from the trained model object |
| `trained_model_K5.pkl` | The locked, trained model (K=5 states) — a Python pickle containing the fitted `hmmlearn.GaussianHMM` object plus feature/label metadata |
| `training_data_daily_ohlcv.csv` | Daily BTC/USDT OHLCV, 2017-08-17 to 2026-07-22 (3,262 rows) — what the HMM was fit on |
| `decoded_daily_regime_states.csv` | Per-day features + the decoded regime (both filtered/causal and smoothed/historical-only probabilities) |
| `regime_plot_visualization.png` | Price chart colored by decoded regime |
| `full_model_comparison_report.txt` | The complete K=2..5 model comparison (BIC, per-state parameters, transition matrix) that justified selecting K=5 |

## The 5 regimes found

| Regime | Mean return/day | Mean vol/day | Mean ADX | Duration |
|---|---|---|---|---|
| Strong bull breakout | +0.64% | 3.3% | 48.2 | ~18.5 days |
| Moderate uptrend | +0.23% | 3.0% | 33.3 | ~12.3 days |
| Chop | +0.02% | 3.5% | 23.0 | ~15.9 days |
| Quiet accumulation | +0.08% | 1.8% | 18.8 | ~23.7 days |
| Bear / crash | −0.57% | 6.7% | 35.0 | ~21.2 days |

## Critical methodology note: filtered vs. smoothed decoding

Standard HMM decoding (Viterbi, `predict_proba`) uses the ENTIRE sequence —
past and future — to assign a state at time t. That's lookahead bias if
used live. This model uses a hand-implemented **causal forward algorithm**
(`filtered_state_probs()` in the training script) for anything live-facing.
The smoothed decode is included in the output CSV only for historical
interpretation — never use the `smoothed_*` columns for anything live.

## Was this ever added to the trading model?

Tested extensively (see `09_Bugs_Found_And_Fixed` and the master
methodology doc) — regime probabilities did not clear the bar for a
statistically robust AUC improvement when added to the CatBoost classifier.
The better-supported use is direct, rule-based position sizing (bear-regime
trades lose ~5.7x more on average), not a model input feature.

## Reproduction

```bash
python train_hmm_model.py --start_date 2017-08-17 --k_min 2 --k_max 5 \
    --n_restarts 20 --features logret,vol,adx --select_k 5
```
