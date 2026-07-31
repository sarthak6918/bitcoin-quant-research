# BTC Regime & Entry-Classifier Project — Complete Package

Everything built and investigated across this project: a Hidden Markov
Model for market regime detection, multiple CatBoost entry classifiers,
a from-scratch rebuilt dataset with every known bug fixed, deep learning
comparisons, and a path-analysis investigation into exit strategies.

**Read `00_METHODOLOGY_AND_FINDINGS.md` next** — it's the full narrative:
what was built, what was tested, what was found, and what the honest
final conclusion is.

## Folder index

| Folder | Contents |
|---|---|
| `01_HMM_Regime_Model/` | The Gaussian HMM: training code, trained model, training data, decoded regime states |
| `02_Master_Hourly_OHLCV_With_Regime/` | The single most-reused dataset: 78,150 hourly BTC/USDT bars with regime mapped on, 2017–2026 |
| `03_CatBoost_Model_A_Deployed_Baseline_10bar/` | The model actually live in production before any HMM work — 10-bar barrier target |
| `04_CatBoost_Model_B_5bar_Variant/` | Research variant testing a shorter barrier — higher historical AUC, worse live AUC |
| `05_Final_Rigorous_Dataset_4123_Signals/` | **The most trustworthy dataset in this project** — built entirely from scratch, no inherited bugs |
| `06_Walkforward_Validation_And_Final_Model/` | Walk-forward validation + final model on the from-scratch dataset |
| `07_Deep_Learning_Experiments/` | MLP and LSTM experiments — tests whether this is a modeling-capacity problem (it isn't) |
| `08_Path_Analysis_Exit_Strategy/` | Investigation into whether confidence-conditioned exit rules could work (they don't, yet) |
| `09_Bugs_Found_And_Fixed/` | Full documentation of 3 real, confirmed bugs found during this investigation |

## The one-paragraph summary

The current entry pattern (StochRSI + ADX crossover) shows historical AUC
of 0.85–0.91 on the original training data, but this figure does not hold
up on genuinely fresh 2026 data (~0.49–0.57 depending on configuration).
Extensive investigation — three real bugs found and fixed, a from-scratch
dataset rebuild, three different model architectures (gradient boosting,
MLP, LSTM), and a path-analysis check on exit strategies — all converge on
the same conclusion: **the original high historical AUC was very likely a
data-construction artifact, not real predictive signal that later decayed.**
A from-scratch rebuild shows only ~0.53 AUC even in pure walk-forward
validation on historical data alone. The path forward is genuinely new
information (order flow, funding rate/OI, cross-asset signals) rather than
further model or feature engineering on the existing inputs.

## Reproducibility

Every trained model in this package (`03_`, `04_`) was re-verified to
reproduce its reported AUC exactly, in a clean directory, before packaging.
The HMM's reproduction command is in `01_HMM_Regime_Model/README.md`.
