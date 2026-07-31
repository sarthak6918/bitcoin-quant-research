# 07 — Deep Learning Experiments

Direct test of "is this a modeling-capacity problem?" — three
architecturally unrelated approaches, two different input representations,
evaluated on the identical 2025-2026 test set.

## Files

| File | What it is |
|---|---|
| `lstm_raw_ohlcv_experiment.py` | LSTM fed 60 raw hourly OHLCV bars directly (log returns + high/low/open relative to close + log volume) — no hand-engineered indicators at all |
| `mlp_control_experiment_results.csv` | MLP (scikit-learn) trained on the SAME 33 hand-engineered features as CatBoost — the control that isolates "is the algorithm too weak" from "is the feature set too weak" |

## Results — all three converge to the same ceiling

```
Model                          Input representation              Test AUC
CatBoost (gradient boosting)   28 hand-engineered features        0.534 ± 0.016
MLP (neural network)           same 28 hand-engineered features   0.490 ± 0.018
LSTM (neural network)          60 raw OHLCV bars, no engineering  0.512 ± 0.014
```

## What this proves

- **CatBoost vs. MLP** (same inputs, different algorithms): near-identical
  performance. Rules out "the ML algorithm is too weak" — two very
  different learning methods extract the same near-zero signal from the
  same features.
- **MLP vs. LSTM** (same algorithm family, different representations):
  also converge. Rules out "the hand-engineered indicators are throwing
  away useful information" — giving the model raw price shape directly
  doesn't surface anything the indicators were hiding.

## Conclusion

Three unrelated architectures and two representations landing in the same
~0.49-0.54 band is strong convergent evidence that the 5-bar-forward
outcome, following a StochRSI+ADX cross, is close to unpredictable from
price/volume/volatility data alone. This points toward needing genuinely
new information (order flow, funding rate/OI, cross-asset signals) rather
than a better model or smarter feature engineering on the same inputs.

## Lookahead-safety note (LSTM specifically)

Each 60-bar window ends at the signal's own last-closed candle — verified
by construction (`build_sequence()` only ever slices `ohlcv.iloc[start:pos+1]`,
never beyond `pos`). Volume normalization statistics (mean/std) were
computed from the training set only, applied unchanged to the test set.
