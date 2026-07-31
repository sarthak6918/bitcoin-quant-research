# Strategy-Input Search: Can Tuning the Pinescript's Own Inputs Produce
# Entries That Correlate With Our Features? — No.

**Ask**: full independence to change the pinescript's tunable inputs (RSI
length, Stochastic length, K/D smoothing, ADX length/threshold, StochRSI
oversold/overbought thresholds) to find a parameterization whose resulting
BUY/SELL entries are maximally correlated with our engineered feature set,
at a modest ~2 signals/day frequency.

## Method

- Operationalized "correlation with our features" concretely: our causal,
  walk-forward, leakage-checked CatBoost model (n=1 bar horizon, folder
  `10_Direction_Prediction_Rebuild`) is literally a function of the feature
  set. Its predicted probability's distance from 0.5 (`conviction`) is a
  direct, honest proxy for "how much do our features have to say about this
  bar." A strategy whose entries land on high-conviction bars is, by
  construction, one whose entries correlate with our features.
- Kept the pinescript's exact logical structure (StochRSI %K/%D crossover +
  prevK oversold/overbought filter + ADX trend-strength filter) — only the
  **inputs** were searched, not the logic, per the brief.
- Grid: RSI length {10,14,21} x Stoch length {10,14,21} x ADX length
  {10,14,21} x ADX threshold {15,20,25,30} x oversold/overbought threshold
  {15,20,25,30} x %K threshold {25,30,35,40} x %D threshold {15,20,25,30} =
  6,912 combinations.
- **Search period**: 2021-05-06 -> 2024-04-08 (causal out-of-fold model
  predictions, `gen_oof_predictions.py`) — every candidate scored here only.
- **Validation period**: 2025-01-01 -> 2026-07-22 — touched exactly once,
  after the winning candidate was already selected from the search period.
- Frequency constraint enforced during search: 1.5-2.5 signals/day
  (combined BUY+SELL) -> 2,916 of 6,912 candidates qualified.

## Result: no parameterization works

```
Across all 2,916 qualifying candidates (search period):
  AUC of strategy-direction-vs-actual-outcome:  0.468 - 0.504  (mean 0.488)
  Agreement with model's own directional call:  0.412 - 0.539  (mean 0.473)
  Mean conviction of fired bars:                 ~0.117 - 0.126

Best-AUC candidate (rsi=10, stoch=14, adx_len=14, adx_th=15,
os_th=25, k_th=25, d_th=20), validated on the untouched 2025-2026 holdout:
  Frequency:        2.02 signals/day  (hit the target)
  AUC:              0.505  (random)
  Agreement rate:   0.485  (slightly BELOW a coin flip)
  Mean conviction of fired bars: 0.102, vs. 0.126 baseline across ALL bars
```

The last line is the most telling: the best candidate this search could
find actually selects entries on bars where our model is **less** confident
than its unconditional average, not more. Across nearly 3,000 tested
parameterizations spanning a wide, reasonable input range, the ceiling for
both AUC and model-agreement sits at essentially 0.50 — indistinguishable
from chance, and this holds up unchanged on the held-out validation period
(no search-period overfitting artifact).

## Why this happens, structurally

StochRSI %K/%D crossover with an ADX trend filter is a **momentum
turning-point** trigger: it fires when short-term momentum reverses inside
an already-trending market. Our model's conviction is driven by a
different, broader mix of signals — multi-lookback return/vol structure,
return autocorrelation, multi-timeframe (4h/daily) trend context, and
candle-position-in-range. There is no structural reason these two things
should line up, and empirically, across the entire searched input space,
they don't. Tuning the *inputs* of this specific rule family cannot fix a
mismatch in what the rule *structurally* looks for versus what the feature
set captures — this is a logic-family problem, not a threshold-tuning
problem.

## What would actually work, if the goal is "entries correlated with our features"

The only strategy that is *by construction* maximally correlated with the
feature set is one gated directly on the model's own output — e.g., fire
BUY when `model_prob >= threshold_high`, SELL when `model_prob <=
threshold_low`, with thresholds chosen to hit ~2 signals/day. This can't be
expressed as a small tweak to the existing pinescript's RSI/Stoch/ADX
inputs (Pine Script can't run a CatBoost model natively), but it can be
deployed as an external signal (Python inference -> TradingView webhook
alert, or a simplified linear proxy of the top features implemented
directly in Pine). Say the word and I'll build whichever of those two you
want.
